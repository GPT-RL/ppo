import os
import re
import subprocess
import sys
import time
from typing import Dict, List


def check_output(command: List[str], suppress_stderr: bool = True) -> str:
    """Runs subprocess.check_output and returns the result as a string.

    :param command: A list of strings representing the command to run on the command line.
    :param suppress_stderr: Whether to suppress anything written to standard error.
    :return: The output of the command, converted from bytes to string and stripped.
    """
    with open(os.devnull, "w") as devnull:
        devnull = devnull if suppress_stderr else None
        output = (
            subprocess.check_output(command, stderr=devnull).decode("utf-8").strip()
        )
    return output


def has_git() -> bool:
    """Returns whether git is installed.

    :return: True if git is installed, False otherwise.
    """
    try:
        output = check_output(["git", "rev-parse", "--is-inside-work-tree"])
        return output == "true"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_git_root() -> str:
    """Gets the root directory of the git repo where the command is run.

    :return: The root directory of the current git repo.
    """
    return check_output(["git", "rev-parse", "--show-toplevel"])


def get_git_hash() -> str:
    """Gets the git hash of HEAD of the git repo where the command is run.

    :return: The git hash of HEAD of the current git repo.
    """
    return check_output(["git", "rev-parse", "HEAD"])


def get_git_url(commit_hash: bool = True) -> str:
    """Gets the https url of the git repo where the command is run.

    :param commit_hash: If True, the url links to the latest local git commit hash.
    If False, the url links to the general git url.
    :return: The https url of the current git repo.
    """
    # Get git url (either https or ssh)
    try:
        url = check_output(["git", "remote", "get-url", "origin"])
    except subprocess.CalledProcessError:
        # For git versions <2.0
        url = check_output(["git", "config", "--get", "remote.origin.url"])

    # Remove .git at end
    url = url[: -len(".git")]

    # Convert ssh url to https url
    m = re.search("git@(.+):", url)
    if m is not None:
        domain = m.group(1)
        path = url[m.span()[1] :]
        url = f"https://{domain}/{path}"

    if commit_hash:
        # Add tree and hash of current commit
        url = f"{url}/tree/{get_git_hash()}"

    return url


def has_uncommitted_changes() -> bool:
    """Returns whether there are uncommitted changes in the git repo where the command is run.

    :return: True if there are uncommitted changes in the current git repo, False otherwise.
    """
    status = check_output(["git", "status"])

    return not status.endswith(NO_CHANGES_STATUS)


_find_unsafe = re.compile(r"[^\w@%+=:,./-]", re.ASCII).search


def quote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"
    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'
    return "'" + s.replace("'", "'\"'\"'") + "'"


def get_reproducibility_info() -> Dict[str, str]:
    """Gets a dictionary of reproducibility information.

    Reproducibility information always includes:
    - command_line: The command line command used to execute the code.
    - time: The current time.

    If git is installed, reproducibility information also includes:
    - git_root: The root of the git repo where the command is run.
    - git_url: The url of the current hash of the git repo where the command is run.
               Ex. https://github.com/swansonk14/rationale-alignment/tree/<hash>
    - git_has_uncommitted_changes: Whether the current git repo has uncommitted changes.

    :return: A dictionary of reproducibility information.
    """
    reproducibility = {
        "command_line": f'python {" ".join(quote(arg) for arg in sys.argv)}',
        "time": time.strftime("%c"),
    }

    if has_git():
        reproducibility["git_root"] = get_git_root()
        reproducibility["git_url"] = get_git_url(commit_hash=True)
        reproducibility["git_has_uncommitted_changes"] = has_uncommitted_changes()

    return reproducibility


NO_CHANGES_STATUS = """nothing to commit, working tree clean"""
