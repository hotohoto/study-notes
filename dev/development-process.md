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

### Trunk based development

- Required for the continuous integration
- Known as used by Google
- References
  - https://trunkbaseddevelopment.com/

#### Branch types

- `main`
  - supposed to be release-ready
  - it's the trunk branch
- development branches
  - merged into `main` always instead of any release branch
  - supposed to be short-lived
    - merged within 2-3 days at most
  - hotfixes are also made in a development branch
    - merged into `main` as always
    - cherry-picked to a release branch
    - this might be difficult when
      - there has been big refactoring in `main`
- release branches
  - not mandatory

#### Tips for making short-lived development branch

- use feature flags
  - but avoid technical depts by removing useless ones later
- change with an abstraction layer
  - to make it procedural

### GitFlow

- The `main` branch refers to a clean release tag.
  - reset with `--hard` after a new release tag of the latest version is made
  - don't merge `develop` into `main`.
- The `develop` branch is for continuous development.
- A `release` branches is a temporary branch for preparing release.
- A `support` branch is for supporting old stable versions when `main` cannot refer to it anymore.
- hotfix
  - made from `main` or `support` and merged back to `develop`
  - this might be difficult when
    - there are frequent hotfixes made
    - there has been big refactoring in `develop`
    - but this is unavoidable

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
