# Publishing pyFIES

A one-time setup walkthrough for getting pyFIES live on GitHub, GitHub Pages,
and PyPI. After this is wired up, future releases need only `git tag v0.x.y &&
git push --tags`.

The repository ships with three GitHub Actions workflows:

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | Push or PR to `main` | Runs tests + lint on Linux/macOS x Python 3.11–3.13. |
| `.github/workflows/docs.yml` | Push to `main` | Builds the mkdocs site and deploys it to GitHub Pages. |
| `.github/workflows/release.yml` | Push of a `v*` tag | Builds the wheel + sdist and publishes to PyPI via Trusted Publishing. Manual `workflow_dispatch` runs publish to TestPyPI instead. |

## 1. Push the repo to GitHub

You said your handle is `nejohnson2`. From the project root:

```bash
# create the repo on github.com first (Settings -> New repository)
# Name: pyFIES, Description: Python implementation of FAO's FIES (SDG 2.1.2),
# Public, no README/license/gitignore (we already have them).

git remote add origin git@github.com:nejohnson2/pyFIES.git
git branch -M main
git push -u origin main
```

The CI workflow will fire automatically and you should see a green check next
to your commit on GitHub within a couple minutes.

## 2. Enable GitHub Pages for docs

Once the first push lands:

1. Go to your repo on GitHub: **Settings → Pages**.
2. Under "Build and deployment", set **Source** to `GitHub Actions`.
3. Push any commit to `main` (or re-run the docs workflow). The site appears
   at `https://nejohnson2.github.io/pyFIES/` within ~1 minute.

The first push of `main` will fail the docs workflow if Pages isn't enabled
yet — that's expected. Just enable it and re-run.

## 3. Set up PyPI publishing (one-time)

We use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no
API tokens, no secrets stored anywhere. PyPI verifies that the publish
request is coming from your GitHub Actions workflow on your specific repo.

### 3a. Create accounts

* [pypi.org](https://pypi.org/account/register/) — for production releases.
* [test.pypi.org](https://test.pypi.org/account/register/) — for trial runs
  before the real thing. Different account, register separately.

Enable 2FA on both. PyPI requires it for new accounts.

### 3b. Add a "pending" trusted publisher on PyPI

Because `pyfies` doesn't exist on PyPI yet, you register the publisher *before*
the first upload:

1. Sign in to [pypi.org](https://pypi.org/) → **Your account → Publishing**.
2. Under **Add a new pending publisher**, fill in:
   * PyPI Project Name: `pyfies`
   * Owner: `nejohnson2`
   * Repository name: `pyFIES`
   * Workflow name: `release.yml`
   * Environment name: `pypi`
3. Click **Add**.

Repeat on [test.pypi.org](https://test.pypi.org/) with environment name
`testpypi`.

### 3c. Create the GitHub environments

GitHub side:

1. Repo → **Settings → Environments → New environment**.
2. Name: `pypi`. (No protection rules needed for now; you can add a manual
   approval later if you want.)
3. Repeat for `testpypi`.

These names must match the `environment.name` fields in `release.yml`.

## 4. Test the release pipeline against TestPyPI

Before the real v0.1.0 release, do a dry run:

1. Repo → **Actions → Release → Run workflow** (use `workflow_dispatch`).
2. The workflow builds the wheel and uploads to TestPyPI.
3. Verify on `https://test.pypi.org/project/pyfies/`.
4. In a clean venv, install from TestPyPI to confirm the wheel works:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ \
               pyfies
   python -c "from pyfies import RaschModel; print('ok')"
   ```

## 5. Cut the v0.1.0 release

When you're ready:

```bash
# Bump the version in pyproject.toml (currently 0.1.0.dev0 -> 0.1.0)
# and src/pyfies/__init__.py (currently 0.1.0.dev0 -> 0.1.0).
# Update CHANGELOG.md: move "Unreleased" entries under a new "## [0.1.0] - 2026-04-27" heading.

git add -A
git commit -m "Release v0.1.0"
git tag -a v0.1.0 -m "pyFIES v0.1.0 — first public release"
git push origin main
git push origin v0.1.0
```

The tag push triggers `release.yml`, which builds and publishes to PyPI. After
~2 minutes `pip install pyfies` works for anyone in the world.

## 6. Optional: GitHub Release notes

After the tag is pushed:

1. Repo → **Releases → Draft a new release**.
2. Choose the `v0.1.0` tag.
3. Title: `pyFIES v0.1.0`.
4. Paste the relevant CHANGELOG.md section as the body.
5. Publish.

This gives the release a permanent, citable URL and adds it to the GitHub
Atom feed users can subscribe to.

## After the first release

For later releases, the loop is:

1. Make changes, merge to `main`. CI keeps it green.
2. Bump the version, update CHANGELOG.
3. `git tag vX.Y.Z && git push --tags`.
4. Done.
