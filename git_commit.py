""" Accessing Git Commit Attributes via Python

    Note that you cannot directly change the commit dates using GitPython.
    You would still need to rewrite the repository history with tools like git-filter-repo or git rebase commands.

    # GIT: change commit date to author date
    git filter-branch --env-filter 'export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"'

    git filter-branch --env-filter 'export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"' -f

    git push --force

    git log --format="%h %ad %s" --date=short

    If you're only inspecting the repository and want to list the current dates,
    the following Python script using GitPython can show commit and author dates:


    pip install GitPython


"""

import git
from datetime import datetime

# Path to your Git repository
repo_path = '.'

# Initialize the repo object
repo = git.Repo(repo_path)
n_commits = len(list(repo.iter_commits()))
# # Get a specific commit by its hash
# commit_hash = 'your_commit_hash_here'
# commit = repo.commit(commit_hash)

for i, commit in enumerate(repo.iter_commits()):
    # Print commit information
    print(f"\n\nCommit {i}/{n_commits} ...")
    print("Commit Hash:", commit.hexsha)
    print("Author:", commit.author.name)
    print("Author Email:", commit.author.email)
    print("Author Date:", datetime.fromtimestamp(commit.authored_date))
    print("Committer:", commit.committer.name)
    print("Committer Email:", commit.committer.email)
    print("Committer Date:", datetime.fromtimestamp(commit.committed_date))
    print("Message:", commit.message)
    print("Parents:", [p.hexsha for p in commit.parents])
    print("Tree:", commit.tree)
