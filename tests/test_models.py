#!/usr/bin/env python3
"""Unit tests for deepresearcher2.models module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepresearcher2.models import (
    DeepState,
    FinalSummary,
    GameResult,
    Reference,
    Reflection,
    WebSearchQuery,
    WebSearchResult,
    WebSearchSummary,
)


class TestDeepState:
    """Tests for the DeepState model."""

    def test_default_values(self) -> None:
        """Test that DeepState has correct default values."""
        state = DeepState()

        assert state.topic == "petrichor"
        assert state.search_query is None
        assert state.search_results is None
        assert state.search_summaries is None
        assert state.reflection is None
        assert state.count == 0

    def test_custom_values(self) -> None:
        """Test DeepState with custom values."""
        query = WebSearchQuery(query="test", aspect="basics", rationale="exploring")
        result = WebSearchResult(title="Test Result", url="https://example.com")
        summary = WebSearchSummary(summary="A test summary.", aspect="overview")
        reflection = Reflection(knowledge_gaps=["gap1"], covered_topics=["topic1"])

        state = DeepState(
            topic="quantum computing",
            search_query=query,
            search_results=[result],
            search_summaries=[summary],
            reflection=reflection,
            count=5,
        )

        assert state.topic == "quantum computing"
        assert state.search_query == query
        assert state.search_results == [result]
        assert state.search_summaries == [summary]
        assert state.reflection == reflection
        assert state.count == 5

    def test_partial_initialization(self) -> None:
        """Test DeepState with only some fields provided."""
        state = DeepState(topic="machine learning", count=3)

        assert state.topic == "machine learning"
        assert state.count == 3
        assert state.search_query is None


class TestWebSearchQuery:
    """Tests for the WebSearchQuery model."""

    def test_valid_query(self) -> None:
        """Test creating a valid WebSearchQuery."""
        query = WebSearchQuery(
            query="what is quantum entanglement",
            aspect="physics fundamentals",
            rationale="understanding basic concepts",
        )

        assert query.query == "what is quantum entanglement"
        assert query.aspect == "physics fundamentals"
        assert query.rationale == "understanding basic concepts"

    def test_missing_required_field_query(self) -> None:
        """Test that missing query field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WebSearchQuery(aspect="test", rationale="test")  # type: ignore[call-arg]

        assert "query" in str(exc_info.value)

    def test_missing_required_field_aspect(self) -> None:
        """Test that missing aspect field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WebSearchQuery(query="test", rationale="test")  # type: ignore[call-arg]

        assert "aspect" in str(exc_info.value)

    def test_missing_required_field_rationale(self) -> None:
        """Test that missing rationale field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WebSearchQuery(query="test", aspect="test")  # type: ignore[call-arg]

        assert "rationale" in str(exc_info.value)


class TestWebSearchResult:
    """Tests for the WebSearchResult model."""

    def test_minimal_result(self) -> None:
        """Test WebSearchResult with only required fields."""
        result = WebSearchResult(title="Test Article", url="https://example.com/article")

        assert result.title == "Test Article"
        assert result.url == "https://example.com/article"
        assert result.summary is None
        assert result.content is None

    def test_full_result(self) -> None:
        """Test WebSearchResult with all fields."""
        result = WebSearchResult(
            title="Complete Article",
            url="https://example.com/complete",
            summary="This is a summary of the article.",
            content="# Article Content\n\nFull markdown content here.",
        )

        assert result.title == "Complete Article"
        assert result.url == "https://example.com/complete"
        assert result.summary == "This is a summary of the article."
        assert result.content == "# Article Content\n\nFull markdown content here."

    def test_missing_title(self) -> None:
        """Test that missing title raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WebSearchResult(url="https://example.com")  # type: ignore[call-arg]

        assert "title" in str(exc_info.value)

    def test_missing_url(self) -> None:
        """Test that missing url raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WebSearchResult(title="Test")  # type: ignore[call-arg]

        assert "url" in str(exc_info.value)


class TestReference:
    """Tests for the Reference model."""

    def test_valid_reference(self) -> None:
        """Test creating a valid Reference."""
        ref = Reference(title="Wikipedia Article", url="https://en.wikipedia.org/wiki/Test")

        assert ref.title == "Wikipedia Article"
        assert ref.url == "https://en.wikipedia.org/wiki/Test"

    def test_missing_title(self) -> None:
        """Test that missing title raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Reference(url="https://example.com")  # type: ignore[call-arg]

        assert "title" in str(exc_info.value)

    def test_missing_url(self) -> None:
        """Test that missing url raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Reference(title="Test")  # type: ignore[call-arg]

        assert "url" in str(exc_info.value)


class TestWebSearchSummary:
    """Tests for the WebSearchSummary model."""

    def test_minimal_summary(self) -> None:
        """Test WebSearchSummary with only required fields."""
        summary = WebSearchSummary(summary="A summary of search results.", aspect="overview")

        assert summary.summary == "A summary of search results."
        assert summary.aspect == "overview"
        assert summary.references is None

    def test_summary_with_references(self) -> None:
        """Test WebSearchSummary with references."""
        refs = [
            Reference(title="Source 1", url="https://source1.com"),
            Reference(title="Source 2", url="https://source2.com"),
        ]
        summary = WebSearchSummary(
            summary="A comprehensive summary with multiple sources.",
            aspect="detailed analysis",
            references=refs,
        )

        assert summary.summary == "A comprehensive summary with multiple sources."
        assert summary.aspect == "detailed analysis"
        assert summary.references is not None
        assert len(summary.references) == 2
        assert summary.references[0].title == "Source 1"

    def test_missing_summary(self) -> None:
        """Test that missing summary raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WebSearchSummary(aspect="test")  # type: ignore[call-arg]

        assert "summary" in str(exc_info.value)

    def test_missing_aspect(self) -> None:
        """Test that missing aspect raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            WebSearchSummary(summary="test")  # type: ignore[call-arg]

        assert "aspect" in str(exc_info.value)


class TestReflection:
    """Tests for the Reflection model."""

    def test_valid_reflection(self) -> None:
        """Test creating a valid Reflection."""
        reflection = Reflection(
            knowledge_gaps=["area1", "area2"],
            covered_topics=["topic1", "topic2", "topic3"],
        )

        assert reflection.knowledge_gaps == ["area1", "area2"]
        assert reflection.covered_topics == ["topic1", "topic2", "topic3"]

    def test_empty_lists(self) -> None:
        """Test Reflection with empty lists."""
        reflection = Reflection(knowledge_gaps=[], covered_topics=[])

        assert reflection.knowledge_gaps == []
        assert reflection.covered_topics == []

    def test_missing_knowledge_gaps(self) -> None:
        """Test that missing knowledge_gaps raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Reflection(covered_topics=["topic1"])  # type: ignore[call-arg]

        assert "knowledge_gaps" in str(exc_info.value)

    def test_missing_covered_topics(self) -> None:
        """Test that missing covered_topics raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Reflection(knowledge_gaps=["gap1"])  # type: ignore[call-arg]

        assert "covered_topics" in str(exc_info.value)


class TestFinalSummary:
    """Tests for the FinalSummary model."""

    def test_valid_summary(self) -> None:
        """Test creating a valid FinalSummary."""
        summary = FinalSummary(summary="This is the final comprehensive summary of the research.")

        assert summary.summary == "This is the final comprehensive summary of the research."

    def test_missing_summary(self) -> None:
        """Test that missing summary raises ValidationError."""
        with pytest.raises(ValidationError):
            FinalSummary()  # type: ignore[call-arg]


class TestGameResult:
    """Tests for the GameResult enum."""

    def test_enum_values(self) -> None:
        """Test that GameResult has expected values."""
        assert GameResult.A == "A"
        assert GameResult.B == "B"

    def test_enum_members(self) -> None:
        """Test that GameResult has exactly two members."""
        members = list(GameResult)
        assert len(members) == 2
        assert GameResult.A in members
        assert GameResult.B in members

    def test_string_comparison(self) -> None:
        """Test that GameResult can be compared to strings."""
        assert GameResult.A == "A"
        assert GameResult.B == "B"

    def test_enum_from_value(self) -> None:
        """Test creating GameResult from string value."""
        result_a = GameResult("A")
        result_b = GameResult("B")

        assert result_a == GameResult.A
        assert result_b == GameResult.B

    def test_invalid_value(self) -> None:
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            GameResult("C")


class TestModelSerialization:
    """Tests for model serialization and deserialization."""

    def test_deep_state_to_dict(self) -> None:
        """Test DeepState serialization to dict."""
        state = DeepState(topic="AI", count=2)
        data = state.model_dump()

        assert data["topic"] == "AI"
        assert data["count"] == 2
        assert data["search_query"] is None

    def test_deep_state_from_dict(self) -> None:
        """Test DeepState deserialization from dict."""
        data = {
            "topic": "neural networks",
            "count": 10,
            "search_query": None,
            "search_results": None,
            "search_summaries": None,
            "reflection": None,
        }
        state = DeepState.model_validate(data)

        assert state.topic == "neural networks"
        assert state.count == 10

    def test_nested_model_serialization(self) -> None:
        """Test serialization of nested models."""
        summary = WebSearchSummary(
            summary="Test summary",
            aspect="testing",
            references=[Reference(title="Ref", url="https://ref.com")],
        )
        data = summary.model_dump()

        assert data["summary"] == "Test summary"
        assert data["aspect"] == "testing"
        assert data["references"] is not None
        assert len(data["references"]) == 1
        assert data["references"][0]["title"] == "Ref"

    def test_nested_model_deserialization(self) -> None:
        """Test deserialization of nested models."""
        data = {
            "summary": "Deserialized summary",
            "aspect": "deserialization",
            "references": [{"title": "Source", "url": "https://source.com"}],
        }
        summary = WebSearchSummary.model_validate(data)

        assert summary.summary == "Deserialized summary"
        assert summary.references is not None
        assert summary.references[0].title == "Source"

    def test_json_round_trip(self) -> None:
        """Test JSON serialization and deserialization round trip."""
        original = WebSearchQuery(
            query="original query",
            aspect="original aspect",
            rationale="original rationale",
        )
        json_str = original.model_dump_json()
        restored = WebSearchQuery.model_validate_json(json_str)

        assert restored == original
