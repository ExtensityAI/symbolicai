from pathlib import Path
from typing import Optional

from git import Repo

from .. import Expression


class RepositoryCloner(Expression):
    """
    A repository cloner class which allows to clone a repository from
    a provided URL. If the repository is already cloned, it checks if
    the repository is up-to-date and updates it if necessary.

    Parameters:
      repo_path (Optional[str]): The path where to clone the repository.
                                 By default it will be at '~/.symai/repos/'.
    """
    def __init__(self, repo_path: Optional[str] = None):
        super().__init__()
        self.repo_dir = Path.home() / '.symai/repos/' if repo_path is None else Path(repo_path)
        if not self.repo_dir.exists():
            self.repo_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, url: str) -> str:
        """
        Clones a repository if not already cloned or checks if the cloned repository
        is up-to-date and updates it if necessary.

        Parameters:
          url (str): The url of the repository to clone.

        Returns:
          str: The root path of the cloned repository.
        """
        repo_name = url.split('/')[-1].replace('.git', '')
        if (self.repo_dir / repo_name).is_dir():
            print(f'Repository {repo_name} already exists. Checking for updates...')
            try:
                repo = Repo(self.repo_dir / repo_name)
                current = repo.head.commit
                repo.remotes.origin.pull()
                if current != repo.head.commit:
                    print(f'Repository {repo_name} updated.')
                else:
                    print(f'Repository {repo_name} is up-to-date.')
            except Exception as e:
                print(f'An error occurred: {e}')
                raise e
        else:
            print(f'Cloning repository {repo_name}...')
            try:
                Repo.clone_from(url, self.repo_dir / repo_name)
                print(f'Repository {repo_name} cloned successfully.')
            except Exception as e:
                print(f'Failed to clone the repository. An error occurred: {e}')
                raise e
        return str(self.repo_dir / repo_name)
