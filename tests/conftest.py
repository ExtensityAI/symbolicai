import pytest
from dotenv import load_dotenv

# NOTE: live engine tests read provider keys from the environment; a local .env
# (gitignored) can supply them — never commit keys to the repo.
load_dotenv()


def pytest_addoption(parser):
    parser.addoption(
        "--engine-api",
        action="store",
        default="mock",
        choices=("mock", "live"),
        help="Engine API mode: mock validates request shapes, live uses configured provider keys.",
    )


def pytest_report_header(config):
    return f"engine-api: {config.getoption('--engine-api')}"


def pytest_configure(config):
    config.addinivalue_line("markers", "engine_live: runs a live provider API request")


@pytest.fixture
def engine_api_mode(request):
    return request.config.getoption("--engine-api")
