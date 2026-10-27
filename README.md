# AI Suite

[![Open your Home Assistant instance and open this repository inside HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Mugga6315&repository=ha-ai-suite&category=integration)

**A local voice assistant for Home Assistant: fast on simple commands, capable on everything else, with a memory of
the household.**

AI Suite is a conversation agent that runs on your own model servers. A small decision model carries out simple
requests itself, faster than a text model. Everything else goes to a larger text model that answers with tools.
Second Brain gives the assistant a memory it keeps as plain markdown in a git repository.

## Features

- **Decision model first.** A small model handles everyday requests on its own, faster than a text model. If it is
  unsure, the request goes to the text model (or to another conversation agent you choose).
- **A text model with tools** for everything else: Home Assistant's own tools plus AI Suite's tool sets.
- **Tool sets that answer, not dump data:** long-term statistics and state history, every calendar at once with
  weekdays, and an Emby media library. The code works out dates, units and totals; the model gets the result.
- **Memory (Second Brain):** the assistant remembers what it is told, keeps it as markdown notes, and commits every
  change to git so it can be read and reverted.
- **Household rules:** rules written in plain words ("Spiel Artist 1 means the band") are compiled for the decision
  model, so it follows them too.
- **Self-improvement:** a background review notes the turns that went wrong and turns repeated mistakes into rules.
- **Reply check:** before a reply is spoken, AI Suite checks it for a wrong language, numbers no tool gave, or raw
  tool-call text, and asks the model once more.
- **Trace export (optional):** a record of every turn can be sent to a trace receiver of your own.

## How a request is answered

1. The decision model answers a few narrow questions about the request: what to do, which device, which room. Each
   answer comes with a confidence.
2. When it is sure, AI Suite carries the request out through Home Assistant and an answer model words the reply
   ("The kitchen light is on").
3. When it is not sure, or the request is not one it handles, the text model answers it with its tools, or the
   fallback agent does.

The decision model handles:

- switching lights, switches, fans, speakers and thermostats on and off, brightness, temperature, covers and their
  position, locks;
- media: play a song, artist, album, playlist or radio station (through Music Assistant), pause, resume, skip,
  volume, mute;
- shopping and to-do lists: add, tick off, remove, read;
- the current state or value of a device or sensor;
- the weather of a coming day or days, what is in the calendar on a day, or when an appointment is or was;
- a sensor's highest, lowest, average or total value on an earlier day, and when a device last changed;
- thanks and greetings; a request to remember something goes to the text model, which files it.

Requests with several commands ("turn off the lights and close the blinds") are split and carried out part by part.

## Requirements

- Home Assistant 2026.10.0 or newer.
- A **decision server** with the `/v1/systemone` decision API, for example llama.cpp's `llama-server` with a
  decision model.
- An **OpenAI-compatible LLM server** for the text models, for example vLLM or llama.cpp's `llama-server`.
- For playing music by name: Music Assistant.

## Installation

1. With HACS: HACS > ⋮ > Custom repositories, add `https://github.com/Mugga6315/ha-ai-suite` as type Integration,
   then download AI Suite. Without HACS: copy `custom_components/ai_suite` into the `custom_components` folder of your
   Home Assistant configuration.
2. Restart Home Assistant.
3. Settings > Devices & services > Add integration > **AI Suite**, and enter the two servers. AI Suite starts with its
   History and Calendar tool sets.
4. On the AI Suite entry, choose **Add conversation agent**. Under Tools > LLM APIs, select **Assist** and **AI Suite**.
5. Select the agent in a voice assistant: Settings > Voice assistants.
6. Optional: add **Memory** on the AI Suite entry, then select **Second Brain** under the agent's Tools > LLM APIs.
   AI Suite raises a repair issue while Memory is set up and no agent uses it.

AI Suite's memory replaces the separate Second Brain integration; remove that one first.

## Configuration

Everything is set up on the AI Suite entry (Settings > Devices & services > AI Suite): the servers on the entry
itself, the agents, tool sets, Memory and its features as parts added to it.

### Servers

Entered when AI Suite is added; changed later with the entry's ⋮ menu > **Reconfigure**. AI Suite saves them only
when both answer.

| Setting | What it does |
|---|---|
| **Decision server URL** | Base URL of the decision server, without a path. The agents pick their decision model from it. |
| **LLM server URL** | OpenAI-compatible server, URL ending in `/v1`. It runs the text models and Second Brain's background jobs. |
| **Switch thinking per request (enable_thinking)** | Default on. AI Suite sends `enable_thinking` in `chat_template_kwargs`: the answer model and the rule compiler run without thinking, the background jobs with it, and each agent's **Thinking** setting applies. Off: for a server that rejects the field; no request sends it then. |

### Conversation agent

Each agent is a conversation agent you can select in a voice assistant. Add as many as you like.

- **Name**: the agent's name in the voice assistant settings.
- **Instructions**: the system prompt the text model gets. Templates are allowed, for example the current time or a
  sensor's state.

**Models**

| Setting | What it does |
|---|---|
| **Text model** | Answers every request the decision model passes on, with the tools below. |
| **Temperature** | How much replies vary. 0 to 0.3 gives steady, short replies. Empty: the server's default. |
| **Decision model first** | Default on. Every request goes to the decision model first. Off: the text model answers everything. |
| **Decision model** | The model on the decision server this agent uses. Empty: the server's only model. |
| **Minimum confidence** | Default 0.5, from 0 to 1. How sure the decision model must be to act on its own. Higher: fewer wrong actions, more requests passed on. |
| **Answer model** | Words the reply after the decision model acted, and splits requests with several commands. A small, fast model is enough. Empty: the text model. |
| **Fallback agent** | Another conversation agent (for example a cloud assistant) that answers what the decision model passes on, in place of the text model. The decision model's own replies are still worded by the answer model. Used only while Decision model first is on. Empty: the text model. |

**Thinking**

| Setting | What it does |
|---|---|
| **Thinking** | Turns the model's thinking on or off, for models that support it. Empty: the server's default. |
| **Reasoning effort** | Low, medium or high, for models that take an effort level. Empty: the server's default. |
| **Thinking token budget** | The most tokens the model may think per request, for servers that support a limit. Empty: no limit. |
| **Send the model's earlier reasoning back** | Default on. In a request with several rounds (call a tool, read the result, answer), the model sees what it thought before. Some thinking models need this, or their reply after a tool call gets lost. |

**Tools**

| Setting | What it does |
|---|---|
| **LLM APIs** | The tool sets the model may use: Assist (control and read devices), AI Suite (its tool sets), Second Brain (memory), and any other integration's. |
| **Tools left out** | Single tools the model does not get. Fewer tools make small models more reliable. The list shows the tools of the saved APIs: after changing the APIs, save and open the settings again. |
| **Allow parallel tool calls** | Default on. The model may call several tools in one round. Off: one tool per round. |

The model sees only entities exposed to voice assistants (Settings > Voice assistants > Expose); the History tools
can read more, see their setting below. Each tool reaches the model under a plain name (`HassTurnOn`, not
`intent__HassTurnOn`), which small models call more reliably. Where an AI Suite tool does a Home Assistant tool's job
better, the model gets only AI Suite's: the calendar tool replaces `get_events`.

**Replies**

| Setting | What it does |
|---|---|
| **Check each reply** | Default on. Checks the reply for a wrong language, numbers no tool gave, or raw tool-call text, and has it written once more. A request in which the model already acted is not retried. |
| **Remove emojis from replies** | Default off. Removes emojis before the reply is spoken. |
| **Earlier exchanges kept** | How many earlier exchanges the model sees. 0: only the current request. Empty: all. |

## Tool sets

AI Suite's own tools, on its LLM API **AI Suite**. Each set is added and removed on the AI Suite entry and works
without Memory. History and Calendar come with AI Suite; Emby is added when wanted. History and Emby have a status
entity. With Memory set up, a tool call the model got wrong is also noted in the store's `failures.md` for
Self-improvement.

### History

`get_statistics` and `get_history` answer from Home Assistant's recorder: long-term statistics (energy used, an
average temperature, the highest day) and when a device changed. They return the answer, not raw rows, and work out
dates such as "yesterday" themselves.

- **Read unexposed entities** (default on): the tools may read any entity Home Assistant records, also those not
  exposed to voice assistants. Off: exposed entities only.

### Calendar

`get_calendar_events` reads every exposed calendar at once, for any range, each entry with its weekday, which models
otherwise get wrong. A calendar can be named by its name or an alias. Calendars are not exposed by default; AI Suite
raises a repair issue while none are.

### Emby

`count_emby`, `search_emby`, `recent_emby`, `now_playing_emby`, `next_up_emby` and `play_emby` search and play an
Emby (or Jellyfin) library.

- **Server URL** and **API key**: both or neither.
- **Android TV players** (optional): a player that is not connected yet gets Kodi started on it before playback.
  Empty: any Android TV player.

## Memory (Second Brain)

Added on the AI Suite entry. The model gets `search_brain`, `read_note`, `add_memory`, `update_memory` and `forget`.
What the assistant is told lands in `memories/<topic>.md` as a timestamped bullet, with the model's reason in the
commit message, so `git log` shows why it was stored.

| Setting | What it does |
|---|---|
| **Store location** | The store's folder. Default: a `second_brain` folder in the Home Assistant configuration folder. Network mounts under `/share` and `/media` are offered too, and a folder that already holds a store is used as it is. The configuration folder itself and `custom_components` are refused. |
| **Background model** | The model for the background jobs (librarian, rule compiler, self-improvement), on the LLM server. |

Every prompt carries the store's core notes, rules and index, each cut at a fixed length (4000, 2000 and 2000
characters) so that a growing store does not slow every request; the model is told when something was cut. Reading
a note returns at most 8000 characters.

### The store

```
CORE.md              context of your own, in every prompt
INDEX.md             generated routing table
memories/            raw notes per topic
  rules.md           household rules, in every prompt
  self_improving.md  rules the self-improvement learned, in every prompt
  not_a_defect.md    behaviour that is fine, never flagged or learned from
failures.md          what went wrong, the self-improvement's input
wiki/                curated pages the librarian writes
log.md               what the nightly passes changed
```

Everything is markdown in a git repository: `git log` shows every change, `git revert` undoes any of them.

### Librarian

Merges the raw notes into curated `wiki/` pages each night, links them with `[[wikilinks]]`, marks superseded facts
and writes the routing the index uses. Has a status entity for the LLM server.

- **Nightly run** (default on) and its **Time** (default 03:00). Off: it runs only through `ai_suite.consolidate`.
- **Thinking effort** (default high).

### Self-improvement

After every turn the text model answers, a background review checks what was asked, which tools ran and what was
answered. A misfire (a wrong claim, an over-broad action, a rule ignored, the wrong language) or a detour (the right
answer with more tool calls than needed) becomes a line in `failures.md`; clean turns write nothing. A request the
decision model carried out itself is reviewed only in a conversation the text model already answered in.

Each night a second pass reads those lines. Only a pattern seen in two or more of them becomes a rule, at most three
a night, written to `memories/self_improving.md`. The pass also retracts learned rules that turn out wrong. Your
`rules.md` is never changed. A bullet in `memories/not_a_defect.md` marks a behaviour as fine: it is not flagged
again, no rule is learned from it, and a learned rule that contradicts it is retracted. That file goes only to the
review, never to the assistant.

- **Thinking effort** (default high).
- **Time of the nightly pass** (default 03:30). `ai_suite.learn` runs it at once.

Each reviewed turn costs one background-model request, run one at a time and never blocking the conversation.

## Trace monitor export

Off by default; set in the entry's options (Settings > Devices & services > AI Suite > **Configure**). A change
reloads the entry, and saving with the export on checks that the receiver answers and accepts the token. When on, AI
Suite sends a record of every turn its agents answer to a receiver of your own: the decision model's questions and
answers, the model requests and replies, tool calls and results, the actions run, the state changes the turn caused
and, for a voice turn, the pipeline's timings.

| Setting | What it does |
|---|---|
| **Send turns to the monitor** | Default off. A receiver that is down does not slow the turns: up to 100 records wait and are retried up to 5 times, then dropped. |
| **Monitor URL** | The receiver's base URL, without a path. Records go to `/api/ingest/turns`. |
| **Ingest token** | Sent as a bearer token. Never logged, redacted in diagnostics. |
| **Send speech audio** | Default off. Also uploads each voice turn's recordings (WAV, at most 2 MiB each). Needs `assist_pipeline: debug_recording_dir:` in `configuration.yaml`. **Each run folder is deleted once its recordings are uploaded.** |

A record holds what was said and the full prompts; the agent's settings go with it, server URLs without credentials.

## Actions

| Action | What it does |
|---|---|
| `ai_suite.consolidate` | Runs the Librarian now. New memories wait until it finishes, up to 10 minutes on a slow model. |
| `ai_suite.learn` | Runs the Self-improvement pass now. |

Both take no fields and return a summary of what changed as `result` (read it with `response_variable`). A run with
nothing to do succeeds and says so. An action fails when AI Suite is not loaded, when its feature (each needs Memory
with a background model) is missing or Memory is off, or when the run changes nothing because the model's plan was
missing or over the limits (for example more than three new rules).

## Troubleshooting

- **AI Suite keeps retrying its setup:** the LLM server does not answer. Home Assistant retries until it is back. A
  decision server that is down does not stop the setup; every request then goes to the text model.
- **System health** (Settings > System > Repairs > ⋮ > System information): shows the tool sets and features, whether
  the LLM server answers, whether the store's folder is there, and whether the Emby server answers.
- **Debug logs:** the entry's ⋮ menu > **Enable debug logging**, or in `configuration.yaml`:

  ```yaml
  logger:
    logs:
      custom_components.ai_suite: debug
  ```

## Known limitations

- One AI Suite entry per Home Assistant; add several agents to it instead.
- The decision model's word lists and the reply's language check cover German and English. In other languages more
  requests go to the text model, and replies are not checked for their language. A language is one module in
  `custom_components/ai_suite/languages/`, filled like `de.py` and `en.py`.
- The store's history uses git. Home Assistant OS, Container and Supervised include it; a Home Assistant Core install
  needs git on its system, otherwise notes are written without commits.

## Removal

1. Settings > Devices & services > AI Suite, ⋮ > **Delete**. This removes its agents, tool sets and features.
2. With HACS: HACS > AI Suite > ⋮ > **Remove**. Without HACS: delete `custom_components/ai_suite`.
3. Restart Home Assistant.

The memory store stays on disk with its git history; delete its folder by hand if it is no longer wanted.

## Acknowledgements

AI Suite builds on the work of others. Thank you to:

- [@skye-harris](https://github.com/skye-harris) for
  [Local OpenAI LLM](https://github.com/skye-harris/hass_local_openai_llm), which showed how a reply and its tool
  calls stream into Home Assistant's chat log and which server quirks to expect. The agent loop learned from it, and
  the agent settings for temperature, emoji removal, earlier exchanges and parallel tool calls follow its options.
- [@joostlek](https://github.com/joostlek) for Home Assistant's
  [OpenRouter](https://github.com/home-assistant/core/tree/dev/homeassistant/components/open_router) integration,
  the base Local OpenAI LLM grew from.
- [@AboveColin](https://github.com/AboveColin) for [HA-Jev](https://github.com/AboveColin/HA-Jev) and
  [@JanOstrowka](https://github.com/JanOstrowka) for
  [typesafe-assist](https://github.com/JanOstrowka/typesafe-assist), which inspired the decision model first with a
  fallback agent for the rest.
- TypeSafe for Jev, its `/v1/systemone` decision API and its [docs](https://docs.typesafe.ai), which describe the
  approach of asking one narrow question at a time.
- Andrej Karpathy for the [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) pattern
  behind the store's layout: raw notes, curated wiki pages, an index and a log.
