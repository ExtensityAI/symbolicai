import logging
from pathlib import Path

from git import Repo

from ..backend.settings import HOME_PATH
from ..symbol import Expression

logger = logging.getLogger(__name__)


class RepositoryCloner(Expression):
    """
    A repository cloner class which allows to clone a repository from
    a provided URL. If the repository is already cloned, it checks if
    the repository is up-to-date and updates it if necessary.

    Parameters:
      repo_path (Optional[str]): The path where to clone the repository.
                                 By default it will be at '~/.symai/repos/'.
    """

    def __init__(self, repo_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.repo_dir = HOME_PATH / "repos/" if repo_path is None else Path(repo_path)
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
        repo_name = url.rsplit("/", maxsplit=1)[-1].replace(".git", "")
        if (self.repo_dir / repo_name).is_dir():
            logger.info("Repository %s already exists. Checking for updates...", repo_name)
            try:
                repo = Repo(self.repo_dir / repo_name)
                current = repo.head.commit
                repo.remotes.origin.pull()
                if current != repo.head.commit:
                    logger.info("Repository %s updated.", repo_name)
                else:
                    logger.info("Repository %s is up-to-date.", repo_name)
            except Exception as e:
                logger.exception("An error occurred")
                raise e from e
        else:
            logger.info("Cloning repository %s...", repo_name)
            try:
                Repo.clone_from(url, self.repo_dir / repo_name)
                logger.info("Repository %s cloned successfully.", repo_name)
            except Exception as e:
                logger.exception("Failed to clone the repository. An error occurred")
                raise e from e
        return str(self.repo_dir / repo_name)
