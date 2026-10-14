<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contribution Guidelines

## AIConfigurator transition scope

AIConfigurator is in a maintenance-only transition to
[AISimulate](https://github.com/ai-dynamo/aisimulate). New features, model or
hardware coverage, and other active development must be proposed and
implemented in AISimulate. The AIC repository accepts only bug, security, and
migration-blocking fixes during its compatibility window.

Documentation corrections and small fixes to existing AIC behavior can be
contributed here by following the rules below. Report security vulnerabilities
through [SECURITY.md](SECURITY.md), not through a public issue or pull request.

Before submitting a significant AIC compatibility or migration fix, open an
[AIC issue](https://github.com/ai-dynamo/aiconfigurator/issues) describing the
problem so the Dynamo team can confirm that it belongs in this frozen
repository rather than AISimulate.

- As part of the GitHub issue discussion, the scope of your fix
  will be agreed upon. An up-front design discussion is required to
  ensure that it preserves the AIC compatibility contract and does not create
  new active-product behavior in this repository.

- The Dynamo project is spread across multiple GitHub repositories.
  The Dynamo team will provide guidance about how and where your change
  should be implemented.

- Testing is a critical part of any Dynamo
  change. You should plan on spending significant time on
  creating tests for your change. The Dynamo team will help you to
  design your testing so that it is compatible with existing testing
  infrastructure.

- If your fix changes user-visible behavior then you need to
  provide documentation.


# Join the Community

Join the community on [Discord](https://discord.gg/mRJ2KNzwYE) to get help, share your ideas, and stay updated on the latest news and developments.

# Contribution Rules

- The code style convention is enforced by common formatting tools
  for a given language (such as clang-format for c++, ruff for python).
  See below on how to ensure your contributions conform. In general please follow
  the existing conventions in the relevant file, submodule, module,
  and project when you add new code or when you extend/fix existing
  functionality.

- Avoid introducing unnecessary complexity into existing code so that
  maintainability and readability are preserved.

- Try to keep code changes for each pull request (PR) as concise as possible:

  - Fillout PR template with clear description and mark applicable checkboxes

  - Avoid committing commented-out code.

  - Wherever possible, each PR should address a single concern. If
    there are several otherwise-unrelated things that should be fixed
    to reach a desired endpoint, it is perfectly fine to open several
    PRs and state in the description which PR depends on another
    PR. The more complex the changes are in a single PR, the more time
    it will take to review those changes.

  - Make sure that the build log is clean, meaning no warnings or
    errors should be present.

  - Make sure all tests pass.

  - Reviewers are auto-requested from the team that owns the areas your
    PR touches (see the generated `CODEOWNERS`). To preview them:
    `python .github/codeowners/who_owns.py --codeowners CODEOWNERS --changed`.
    If the `codeowners` check fails because your PR adds a directory no
    area claims, add a one-line claim in `.github/codeowners/areas.yaml`,
    regenerate, and commit both files (see `.github/codeowners/README.md`
    or the `aic-codeowners` skill). Never edit `CODEOWNERS` directly - it
    is generated.


- Make sure that you can contribute your work to open source (no
  license and/or patent conflict is introduced by your code).
  You must certify compliance with the
  [license terms](https://github.com/ai-dynamo/aiconfigurator/blob/main/LICENSE)
  and sign off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org)
  described below before your pull request (PR) can be merged.

- Thanks in advance for your patience as we review your contributions;
  we do appreciate them!

# Developer Certificate of Origin

Dynamo is an open source product released under
the Apache 2.0 license (see either
[the Apache site](https://www.apache.org/licenses/LICENSE-2.0) or
the [LICENSE file](./LICENSE)). The Apache 2.0 license allows you
to freely use, modify, distribute, and sell your own products
that include Apache 2.0 licensed software.

We respect intellectual property rights of others and we want
to make sure all incoming contributions are correctly attributed
and licensed. A Developer Certificate of Origin (DCO) is a
lightweight mechanism to do that.

The DCO is a declaration attached to every contribution made by
every developer. In the commit message of the contribution,
the developer simply adds a `Signed-off-by` statement and thereby
agrees to the DCO, which you can find below or at [DeveloperCertificate.org](http://developercertificate.org/).

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

We require that every contribution to Dynamo is signed with
a Developer Certificate of Origin. Additionally, please use your real name.
We do not accept anonymous contributors nor those utilizing pseudonyms.

Each commit must include a DCO which looks like this

```
Signed-off-by: Jane Smith <jane.smith@email.com>
```
You may type this line on your own when writing your commit messages.
However, if your user.name and user.email are set in your git configs,
you can use `-s` or `--signoff` to add the `Signed-off-by` line to
the end of the commit message.
