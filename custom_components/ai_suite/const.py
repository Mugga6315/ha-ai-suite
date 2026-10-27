"""Constants for AI Suite."""

DOMAIN = "ai_suite"

# The config entry: the servers every agent of the home shares.
CONF_DECIDER_URL = "decider_url"  # the /v1/systemone decision server every agent uses
# One of the decision server's models, sent with each request; absent: the server's only model.
CONF_DECIDER_MODEL = "decider_model"
CONF_LLM_URL = "llm_url"  # an OpenAI-compatible server (vLLM, llama.cpp, llama-swap)
# Off for servers that reject chat_template_kwargs (vLLM with --tokenizer-mode mistral answers 400).
CONF_THINKING_OFF_KWARG = "thinking_off_kwarg"

# A conversation subentry: one conversation agent.
SUBENTRY_CONVERSATION = "conversation"
CONF_DECIDER = "decider"  # the decision model goes first; off: the agent loop answers every turn
DEFAULT_DECIDER = True
CONF_FALLBACK_AGENT = "fallback_agent"  # takes the turns the decider passes on; absent: the agent loop
CONF_THRESHOLD = "threshold"
# The loop's reply is checked before the user hears it, and written once more on a fault (agent/reply_check.py).
CONF_REPLY_CHECK = "reply_check"
CONF_ANSWER_MODEL = "answer_model"  # words decided turns and splits several requests; absent: the loop's model

# The platform the loop's tool calls run intents under, apart from the decider's own (DOMAIN): HA only names it in
# the intent (intent.async_handle), and logs and traces tell the two apart by it.
AGENT_PLATFORM = f"{DOMAIN}.agent"

DEFAULT_THRESHOLD = 0.5
# On: it removes made-up values without harming right replies; it costs one more round only on a flagged reply.
DEFAULT_REPLY_CHECK = True
DEFAULT_THINKING_OFF_KWARG = True
TIMEOUT_SECONDS = 8
