name: Pull Request Template
description: Template for submitting a new pull request for the project.
title: "[PR] Title of the PR"
labels: []
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        ## 🚀 Pull Request Template
        Please provide a detailed description of the changes introduced by this pull request.

  - type: input
    id: pr_title
    attributes:
      label: PR Title
      description: Provide a brief title for the pull request.
      placeholder: "Enter PR title"
    validations:
      required: true

  - type: textarea
    id: pr_description
    attributes:
      label: Description
      description: Describe the purpose and details of the pull request. Include the reasoning behind the changes.
      placeholder: "Enter detailed description of the changes"
    validations:
      required: true

  - type: input
    id: related_issues
    attributes:
      label: Related Issues
      description: List any related issues or tasks this PR addresses (e.g., #123).
      placeholder: "e.g., #123"
    validations:
      required: false

  - type: input
    id: testing_instructions
    attributes:
      label: Testing Instructions
      description: Describe how to test the changes made in this PR.
      placeholder: "Enter testing instructions"
    validations:
      required: true

  - type: input
    id: impact
    attributes:
      label: Impact
      description: Describe the potential impact of these changes on the project.
      placeholder: "Enter impact description"
    validations:
      required: false

  - type: input
    id: documentation_update
    attributes:
      label: Documentation Update
      description: Did you update the documentation? (e.g., README, API docs, etc.)
      placeholder: "Yes/No"
    validations:
      required: true
