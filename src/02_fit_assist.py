from fit_assist_common import run_fit_assist


def main():
    run_fit_assist("mcmc", include_pybkt=True)
    for method in ["mle", "vb", "pathfinder"]:
        run_fit_assist(method)


if __name__ == "__main__":
    main()
