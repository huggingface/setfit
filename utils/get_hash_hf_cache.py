import hashlib

import huggingface_hub


def get_all_commit_hashes(revisions):
    """Get all commit hashes from a sequence of repo revisions."""
    return sorted([revision.commit_hash for revision in revisions])


def create_repo_unique_str(repo, sep="-"):
    """Create a unique string that describes a repo.

    The string is composed by concatenating the repo_id and all the commit hashes.
    """
    return f"{repo.repo_id}{sep}{''.join(get_all_commit_hashes(repo.revisions))}"


def create_hash_hf_cache(sep="-"):
    """Create a hash of the current state of the HuggingFace cache.

    The hash is computed by concatenating the unique strings of all the models repos in the cache.
    """
    repos_unique_str = sep.join(
        sorted(
            create_repo_unique_str(repo)
            for repo in huggingface_hub.scan_cache_dir().repos
            if repo.repo_type == "model"
        )
    )
    return hashlib.sha256(repos_unique_str.encode()).hexdigest()


def main():
    """Print to stdout the hash of the HuggingFace cache."""
    print(create_hash_hf_cache())


if __name__ == "__main__":
    main()
