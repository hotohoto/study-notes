# Development process

## Intuitions

- Having a product owner or business analyst is quite important for long term success
- Keeping heros as heros may indicate:
  - it's difficult for the others to contribute
  - we need to improve processes

## Good practices

- High-level designs
  - modularization
  - scalability
    - not scalable company/product may not survive after a few/several years
- Low-level designs
  - clean code
  - linting
  - unit test
- Spend enough time defining concepts and terms for everybody to be in the same page
- Make processes shared and improve it
  - business requirements document
  - software requirements document
  - Define best practices
    - Keep documents in the source code
      - README.md
      - how to contribute
      - what is clean architecture
      - patterns
      - anti patterns
      - style guide
    - Use linters and formatters
  - quick PoC
    - Stay agile and responsive
    - Focus on the real values.
    - UI usually not required.
    - Be ready to suggest projects and that's enough.
- Review codes
  - making sure the code quality
  - following best practices together
- Make a template for mock-up
- Make a storyboard
  - using the mock-up the template
  - important to better communication and better test
- Make it everything transparent
- Make people self motivated

## On-boarding

- Get daily feedbacks from the new members.
  - Respect them.
  - Ask what they need.
- Investigate things they've done before.
- Provide hospitality.
- Listen carefully and more
  - they would not accept any advices if they don't agree

## branching workflows

- GitFlow
  - The `main` branch refers to a clean release tag.
    - reset with `--hard` after a new release tag of the latest version is made
    - don't merge `develop` into `main`.
  - The `develop` branch is for continuuous development.
  - A `release` branches is a temporary branch for preparing release.
  - A `support` branch is for supporting old stable versions when `main` cannot refer to it anymore.
  - hotfix
    - made from `main` or `support` and merged back to `develop`
    - this might be difficult when
      - there are frequent hotfixes made
      - there has been big refactoring in `develop`
      - but this is unavoidable

- trunk based developement
  - Known as used by Google
  - The `trunk` branch is for continuous development
  - A `release` branch is for the release tags of a single release.
  - Hotfixes are made in the development branch and cherry-picked to a `release` branch
    - this might be difficult when
      - there are frequent hotfixes made
      - there has been big refactoring in `develop`
    - hotfixed are never merged back to the development branch
  - personal ideas
    - Optionally, we could have `main` for referring to the lateset clean release tag.
      - Then, it would be more similar to `GitFlow`
    - I don't see the big improvements of using the turnk based workflow than GitFlow.
      - In GitFlow, we don't work on `release` that much except for fixing bugs.
      - Maybe because we have less developers??
      - TODO read https://trunkbaseddevelopment.com/ carefully

## Release cycle

e.g.

- develop features
- freeze features
- fix issues which were known during development
- test
  - finds bugs which should've been considered during designing phase for each feature
  - finds other bugs
- fix both kinds of bugs
- finds additional bugs
- fix bugs
- finds additional bugs
- fix bugs
- release
- more bugs reported
- fix bugs
- more bugs reported
- fix bugs
- think it's stable
- more bugs reported
- fix bugs

## References

- https://devblogs.microsoft.com/premier-developer/introduction-to-cmmi-development-for-developers/
- https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
- https://trunkbaseddevelopment.com/
