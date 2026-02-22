from __future__ import annotations

import argparse
import os
from pathlib import Path

from ragbook.eval import run_eval
from ragbook.index import build_and_persist_index, load_index
from ragbook.llm_ollama import OllamaClient
from ragbook.prompt import build_answer_prompt
from ragbook.retrieve import format_citation, hybrid_retrieve
from ragbook.utils import LOGGER, configure_logging


def _default_top_k() -> int:
    try:
        return int(os.getenv("RAG_TOP_K", "6"))
    except ValueError:
        return 6


def cmd_ingest(args: argparse.Namespace) -> None:
    build_and_persist_index(
        pdf_paths=[Path(p) for p in args.pdf],
        out_dir=Path(args.out),
        embed_model=args.embed_model,
    )
    print(f"Index created at: {args.out}")


def cmd_ask(args: argparse.Namespace) -> None:
    idx = load_index(Path(args.index), embed_model_override=args.embed_model)
    top_k = args.top_k if args.top_k is not None else _default_top_k()
    retrieved = hybrid_retrieve(
        idx,
        query=args.q,
        top_k=top_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )

    if args.retrieval_only:
        print("Retrieved passages:\n")
        for r in retrieved:
            c = format_citation(r.chunk)
            section = r.chunk.get("section_title") or "N/A"
            doc = r.chunk.get("book_title") or "N/A"
            print(
                f"{r.rank}. {c} | {r.chunk['chunk_id']} | doc={doc} | section={section} | score={r.hybrid_score:.3f}"
            )
            print(r.chunk["text"])
            print("-" * 80)
        return

    llm = OllamaClient.create(model_override=args.ollama_model)
    prompt = build_answer_prompt(args.q, retrieved)
    answer = llm.generate(prompt)

    print(answer)
    print("\nSources:")
    for r in retrieved:
        c = format_citation(r.chunk)
        section = r.chunk.get("section_title") or "N/A"
        doc = r.chunk.get("book_title") or "N/A"
        print(f"- {c} | {r.chunk['chunk_id']} | doc={doc} | section={section}")


def cmd_eval(args: argparse.Namespace) -> None:
    run_eval(
        index_dir=Path(args.index),
        eval_jsonl=Path(args.eval),
        out_csv=Path(args.out),
        top_k=args.top_k if args.top_k is not None else _default_top_k(),
    )
    print(f"Saved eval results to: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragbook", description="Fusion textbook RAG (local, Ollama)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDF and build hybrid index")
    p_ingest.add_argument("--pdf", nargs="+", required=True, help="One or more input PDF paths")
    p_ingest.add_argument("--out", required=True, help="Output index directory")
    p_ingest.add_argument("--embed_model", default=None, help="Optional embedding model override")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against an existing index")
    p_ask.add_argument("--index", required=True, help="Index directory")
    p_ask.add_argument("--q", required=True, help="Question")
    p_ask.add_argument("--top_k", type=int, default=None, help="Top-k chunks to retrieve")
    p_ask.add_argument("--retrieval_only", action="store_true", help="Only print retrieved passages")
    p_ask.add_argument("--dense_weight", type=float, default=0.65, help="Dense retrieval weight")
    p_ask.add_argument("--sparse_weight", type=float, default=0.35, help="Sparse retrieval weight")
    p_ask.add_argument("--ollama_model", default=None, help="Optional Ollama model override")
    p_ask.add_argument("--embed_model", default=None, help="Optional embedding model override")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Run eval JSONL and save CSV metrics")
    p_eval.add_argument("--index", required=True, help="Index directory")
    p_eval.add_argument("--eval", required=True, help="Eval JSONL path")
    p_eval.add_argument("--out", required=True, help="Output CSV path")
    p_eval.add_argument("--top_k", type=int, default=None, help="Top-k for retrieval")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    LOGGER.debug("args=%s", args)
    args.func(args)


if __name__ == "__main__":
    main()
