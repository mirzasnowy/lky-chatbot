# Example chats

Real transcripts produced by running this system (question, retrieved chunks
with scores, model answer, model id, timestamp). Nothing here is invented:
each transcript is one run of `src/rag.py` or the API, saved verbatim.

To generate: set OPENROUTER_API_KEY in `.env`, then run the questions in
`eval/questions.json` (or your own) through the system and save each
transcript as `examples/chat-NN-slug.json`. `web/chats.json` is the demo
rendering of these transcripts.
