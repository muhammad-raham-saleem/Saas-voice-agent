"""Quick test script for TechFlow RAG engine."""

from src.rag_engine import RAGEngine


def main():
    print("=" * 60)
    print("  TechFlow RAG Engine - Test Suite")
    print("=" * 60)

    engine = RAGEngine()
    stats = engine.get_stats()
    print(f"\n📊 Engine Stats: {stats}")

    test_questions = [
        "How do I reset my password?",
        "What's included in the Pro plan?",
        "What are the API rate limits?",
        "My tasks aren't syncing",
        "What's your refund policy?",
    ]

    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'━' * 60}")
        print(f"  Test {i}/{len(test_questions)}")
        print(f"{'━' * 60}")
        print(f"❓ Question: {question}")

        result = engine.query(question)

        print(f"💬 Answer: {result['answer']}")
        print(f"⏱️  Response time: {result['response_time']}s")
        print(f"📎 Sources: {', '.join(s['source'] for s in result['sources'])}")

        passed = len(result["answer"]) > 20 and len(result["sources"]) > 0
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}")
        results.append(passed)

    print(f"\n{'=' * 60}")
    passed_count = sum(results)
    total = len(results)
    print(f"  Results: {passed_count}/{total} tests passed")
    if passed_count == total:
        print("  ✅ All tests passed!")
    else:
        print("  ⚠️  Some tests failed — review answers above")
    print("=" * 60)


if __name__ == "__main__":
    main()
