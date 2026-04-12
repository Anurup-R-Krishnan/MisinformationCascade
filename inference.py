"""Root submission entrypoint for hackathon evaluators.

Delegates to the package implementation in `misinformation_cascade_env.inference`.
"""

from misinformation_cascade_env.inference import main


if __name__ == "__main__":
    main()
