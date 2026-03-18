# Community

LeLabo is designed to grow as a shared research ecosystem, not just as a personal tool.

## The idea

Research methods are often implemented once, for one paper, in one repository. After that, they are hard to reuse — the setup is opaque, the dependencies are baked in, and running the method outside its original context requires reverse-engineering the whole codebase.

Capsules are designed to change that. A capsule can hold a complete method implementation — model, update rule, config, tests — in a form that travels cleanly between projects and researchers.

The goal is to make it easier for:

- **Authors** to share a working, runnable implementation alongside a paper
- **Researchers** to reuse a baseline without rebuilding it from scratch
- **Reviewers** to run a method on a different dataset or setting
- **Anyone** to compare methods fairly under the same runtime conditions

This is still a direction, not a finished ecosystem. The primitives are there. The community around them is being built.

## Contributing

Contributions are welcome across the full project:

- **Runtime** — improvements to the supervised training loop, callbacks, metrics, or output types
- **Capsules** — new built-in models, update rules, or datasets
- **Documentation** — corrections, examples, new guides
- **Paper packs** — capsule implementations of published methods

To contribute, open an issue or pull request on [GitHub](https://github.com/lelabo-ai/lelabo).

For significant changes, open an issue first to discuss the direction before writing code.

## Sharing capsules

If you have implemented a method as a capsule and want to share it:

1. Make sure it runs cleanly from a fresh install
2. Include a `README.md` that says what the method does and what it claims to reproduce
3. Include at least one smoke test
4. Pack it: `lelabo capsule pack --from outputs/my_run`
5. Share the bundle or open a pull request to add it to the community capsule list

Be honest about what is implemented and what is not. A partial implementation that documents its limits is more useful than a silent one.

## Reporting issues

Open an issue on [GitHub](https://github.com/lelabo-ai/lelabo/issues). Include:

- LeLabo version (`pip show lelabo`)
- Python version
- The command you ran or the code that failed
- The full error output
