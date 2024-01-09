from enum import Enum
from typing import Literal

import marvin.v2

from tests.utils import pytest_mark_class

Sentiment = Literal["Positive", "Negative"]


@marvin.v2.classifier
class GitHubIssueTag(Enum):
    BUG = "bug"
    FEATURE = "feature"
    ENHANCEMENT = "enhancement"
    DOCS = "docs"


@pytest_mark_class("llm")
class TestClassifier:
    class TestSimple:
        def test_classify_bug_tag(self):
            result = GitHubIssueTag("This is a bug")
            assert result == GitHubIssueTag.BUG

        def test_classify_feature_tag(self):
            result = GitHubIssueTag("This is a great feature")
            assert result == GitHubIssueTag.FEATURE

        def test_classify_enhancement_tag(self):
            result = GitHubIssueTag("This is an enhancement")
            assert result == GitHubIssueTag.ENHANCEMENT

        def test_classify_docs_tag(self):
            result = GitHubIssueTag("This is a documentation update")
            assert result == GitHubIssueTag.DOCS

    class TestInstructions:
        @marvin.v2.classifier(instructions="Everything is a bug, no matter what.")
        class GitHubIssueTagInstructions(Enum):
            BUG = "bug"
            FEATURE = "feature"
            ENHANCEMENT = "enhancement"
            DOCS = "docs"

        def test_classify_bug_tag(self, gpt_4):
            result = self.GitHubIssueTagInstructions("This is a great feature!")
            assert result == self.GitHubIssueTagInstructions.BUG

    class TestDocstring:
        @marvin.v2.classifier
        class GitHubIssueTagDocstring(Enum):
            """Everything is a bug, no matter what."""

            BUG = "bug"
            FEATURE = "feature"
            ENHANCEMENT = "enhancement"
            DOCS = "docs"

        def test_classify_bug_tag(self, gpt_4):
            result = self.GitHubIssueTagDocstring("This is a great feature!")
            assert result == self.GitHubIssueTagDocstring.BUG
