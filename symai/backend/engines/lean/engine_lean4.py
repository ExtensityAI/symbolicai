import subprocess
import tempfile
from pathlib import Path
from typing import Any

import docker
import paramiko

from ....symbol import Result
from ....utils import UserMessage
from ...base import Engine


class LeanResult(Result):
    """
    Represents the result of executing a Lean code snippet.

    Attributes:
        _value (Dict[str, str]): A dictionary containing the output of the Lean execution.
    """
    def __init__(self, value: dict[str, str]) -> None:
        """
        Initializes a new LeanResult instance.

        Args:
            value (Dict[str, str]): The result output of the Lean code execution.
        """
        super().__init__(value)
        self._value = value

class LeanEngine(Engine):
    """
    Engine for executing Lean code within a Docker container, providing SSH access for execution.

    Attributes:
        ssh_host (str): The SSH host, defaulting to 'localhost'.
        ssh_port (int): The SSH port, defaulting to 2222.
        ssh_user (str): The SSH username, defaulting to 'root'.
        ssh_key_path (str): The path to the SSH private key, defaulting to '~/.ssh/id_rsa'.
        docker_client (docker.DockerClient): The Docker client used to manage containers.
        container (docker.models.containers.Container): The Docker container used for executing Lean code.
    """

    def __init__(
        self,
        ssh_host: str = 'localhost',
        ssh_port: int = 2222,
        ssh_user: str = 'root',
        ssh_key_path: str = '~/.ssh/id_rsa'
    ) -> None:
        """
        Initializes the LeanEngine with SSH and Docker configurations.

        Args:
            ssh_host (str): The SSH host, defaulting to 'localhost'.
            ssh_port (int): The SSH port, defaulting to 2222.
            ssh_user (str): The SSH username, defaulting to 'root'.
            ssh_key_path (str): The path to the SSH private key, defaulting to '~/.ssh/id_rsa'.
        """
        super().__init__()
        self.ssh_host: str = ssh_host
        self.ssh_port: int = ssh_port
        self.ssh_user: str = ssh_user
        self.ssh_key_path: Path = Path(ssh_key_path).expanduser()
        self.docker_client: docker.DockerClient = docker.from_env()
        self.container: docker.models.containers.Container = self._ensure_container()
        self.name = self.__class__.__name__
        self._setup_ssh()

    def id(self) -> str:
        """
        Returns the identifier for the engine.

        Returns:
            str: The identifier of the LeanEngine, 'lean4'.
        """
        return 'lean4'

    def _ensure_container(self) -> docker.models.containers.Container:
        """
        Ensures the Docker container for Lean execution exists, creating it if necessary.

        Returns:
            docker.models.containers.Container: The Docker container instance used for Lean code execution.
        """
        container_name: str = "lean-container"

        try:
            existing_container: docker.models.containers.Container = self.docker_client.containers.get(container_name)
            existing_container.remove(force=True)
        except docker.errors.NotFound:
            UserMessage(f"No existing container named '{container_name}' found. Proceeding to create a new one.")

        dockerfile: str = """
        FROM buildpack-deps:buster

        ENV ELAN_HOME=/usr/local/elan \
            PATH=/usr/local/elan/bin:$PATH \
            LEAN_VERSION=leanprover/lean4:nightly

        RUN apt-get update && apt-get install -y openssh-server curl && rm -rf /var/lib/apt/lists/*

        RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain $LEAN_VERSION; \
            elan default $LEAN_VERSION; \
            elan --version; \
            lean --version; \
            leanc --version; \
            lake --version;

        RUN mkdir /var/run/sshd && echo 'root:root' | chpasswd && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

        EXPOSE 22
        CMD ["/usr/sbin/sshd", "-D"]
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as temp_dockerfile:
            temp_dockerfile.write(dockerfile)
            dockerfile_path = Path(temp_dockerfile.name)

        image: docker.models.images.Image
        image, _ = self.docker_client.images.build(path=str(dockerfile_path.parent), dockerfile=str(dockerfile_path), tag="lean4-container-image")
        dockerfile_path.unlink()

        container: docker.models.containers.Container = self.docker_client.containers.run(
            image.id,
            detach=True,
            name=container_name,
            ports={'22/tcp': self.ssh_port}
        )
        return container

    def _setup_ssh(self) -> None:
        """
        Sets up SSH access to the Docker container, including generating an SSH key pair if necessary,
        and configuring the container to accept SSH connections using the generated key.
        """
        if not self.ssh_key_path.exists():
            subprocess.run(['ssh-keygen', '-t', 'rsa', '-b', '2048', '-f', str(self.ssh_key_path), '-N', ''], check=True)

        subprocess.run(['docker', 'exec', self.container.id, 'mkdir', '-p', '/root/.ssh'], check=True)
        public_key_path = self.ssh_key_path.parent / f'{self.ssh_key_path.name}.pub'
        subprocess.run(['docker', 'cp', str(public_key_path), f'{self.container.id}:/root/.ssh/authorized_keys'], check=True)
        subprocess.run(['docker', 'exec', self.container.id, 'chmod', '600', '/root/.ssh/authorized_keys'], check=True)
        subprocess.run(['docker', 'exec', self.container.id, 'chown', 'root:root', '/root/.ssh/authorized_keys'], check=True)

    def forward(self, argument: Any) -> tuple[list[LeanResult], dict]:
        """
        Executes Lean code provided as a string or as an object property.

        Args:
            argument (Any): The Lean code to be executed, either as a string or wrapped in an object.

        Returns:
            Tuple[List[LeanResult], dict]: A tuple containing the result of the Lean execution and associated metadata.
        """
        code: str = argument if isinstance(argument, str) else argument.prop.prepared_input

        rsp: LeanResult | None = None
        err: str | None = None
        tmpfile_path: Path | None = None
        metadata: dict[str, Any] = {}
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".lean") as tmpfile:
                tmpfile.write(code.encode())
                tmpfile_path = Path(tmpfile.name)

            output, exec_metadata = self._execute_lean(tmpfile_path)
            metadata.update(exec_metadata)

            if output:
                rsp = LeanResult({'output': output})
            else:
                metadata['status'] = 'no_output'
        except Exception as e:
            err = str(e)
            metadata.update({'status': 'error', 'message': err})
            UserMessage(f"Error during Lean execution: {err}")
        finally:
            if tmpfile_path and tmpfile_path.exists():
                tmpfile_path.unlink()
            if self.container:
                UserMessage(f"Killing Docker container '{self.container.id}'...")
                self.container.remove(force=True)

        return [rsp] if rsp else [], metadata

    def _execute_lean(self, filepath: str) -> tuple[str, dict]:
        """
        Executes a Lean script within the Docker container via SSH.

        Args:
            filepath (str): The path to the Lean file to be executed.

        Returns:
            Tuple[str, dict]: The output from the Lean execution and associated status metadata.
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.ssh_host, port=self.ssh_port, username=self.ssh_user, key_filename=str(self.ssh_key_path))

            elan_path: str = "/usr/local/elan/bin/elan"
            lean_path: str = "/usr/local/elan/bin/lean"

            _stdin, stdout, stderr = ssh.exec_command(f"{elan_path} default stable && {lean_path} --version")
            output: str = stdout.read().decode()
            error: str = stderr.read().decode()
            UserMessage(f"SSH Command Output: {output}")
            UserMessage(f"SSH Command Error: {error}")

            sftp = ssh.open_sftp()
            remote_path: str = f"/root/{Path(filepath).name}"
            sftp.put(filepath, remote_path)
            sftp.close()

            _stdin, stdout, stderr = ssh.exec_command(f"{lean_path} {remote_path}")
            output = stdout.read().decode()
            error = stderr.read().decode()

            ssh.exec_command(f"rm {remote_path}")
            ssh.close()

            if "error" in output.lower() or "error" in error.lower():
                return output, {'status': 'failure'}
            if not output and not error:
                return "Lean program halted successfully with no output.", {'status': 'success'}
            return output, {'status': 'success'}

        except Exception as e:
            UserMessage(f"SSH command execution failed: {e!s}", raise_with=RuntimeError)

    def prepare(self, argument: Any) -> None:
        """
        Prepares the input for Lean execution by processing and converting it into the appropriate format.

        Args:
            argument (Any): The input to be processed and prepared.
        """
        argument.prop.prepared_input = str(argument.prop.processed_input)
