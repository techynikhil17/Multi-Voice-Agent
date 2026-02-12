from dotenv import load_dotenv
import os
from livekit import agents, rtc, api
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.agents.llm import function_tool, ChatContext
from livekit.agents.job import get_job_context
from livekit.plugins import (
    openai,
    elevenlabs,
    deepgram,
    silero,
    noise_cancellation
)

load_dotenv(".env.local")

# --- GLOBAL PROVIDERS ---
LOCAL_LLM = openai.LLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

DEEPGRAM_STT = deepgram.STT(model="nova-3", language="en")
VAD_MODEL = silero.VAD.load()

# --- PRE-INITIALIZED PERSONA TTS ---
TTS_NICK = elevenlabs.TTS(voice_id="TX3LPaxmHKxFdv7VOQHJ")
TTS_RAJU = elevenlabs.TTS(voice_id="JBFqnCBsd6RMkjVDRZzb")
TTS_CHUTKI = elevenlabs.TTS(voice_id="Xb7hH8MSUJpSbSDYk0k2")

# --- PERSONA INSTRUCTIONS ---
NICK_INSTRUCTIONS = (
    "You are a voice AI assistant named Nick. You are STRICTLY a router. "
    "You MUST ONLY respond in English. NEVER output Chinese, Hindi, or any non-English text. "
    "CRITICAL ROUTING RULES — you MUST follow these with ZERO exceptions: "
    "1. If the user mentions ANY technical issue, problem, device not working, error, bug, "
    "or needs help fixing ANYTHING — you MUST call call_support_agent IMMEDIATELY. "
    "You are FORBIDDEN from troubleshooting, diagnosing, suggesting fixes, or asking "
    "clarifying questions about the issue. Just call the tool. "
    "2. If the user mentions wanting to book, schedule, or make an appointment — "
    "you MUST call call_booking_agent IMMEDIATELY. Do NOT handle the booking yourself. "
    "3. If the user says goodbye, 'end conversation', or wants to stop — "
    "call end_conversation to disconnect. "
    "4. If the user just says hello, hi, or asks a general non-technical question, "
    "respond normally and conversationally. "
    "REMEMBER: You are a ROUTER, not a technician. NEVER give technical advice. "
    "Do NOT call call_nick — you ARE Nick. "
)

RAJU_INSTRUCTIONS = (
    "You are a support voice AI assistant named Raju. "
    "You MUST ONLY respond in English. NEVER output Chinese, Hindi, or any non-English text. "
    "The user is facing an issue with: {topic}. "
    "Help the user resolve their technical issue step by step. "
    "MANDATORY POST-RESOLUTION RULE — you MUST follow this EVERY time: "
    "When the user indicates the issue is resolved, fixed, working now, or thanks you, "
    "you MUST do ALL of the following in order: "
    "Step 1: Acknowledge that the issue is resolved (e.g., 'Great, glad that's fixed!'). "
    "Step 2: Ask EXACTLY this question: 'Would you like me to connect you back to Nick?' "
    "Step 3: STOP and WAIT for the user to respond. Do NOT call any tool yet. "
    "After the user responds: "
    "- If they say yes, sure, okay, or anything affirmative → call call_nick. "
    "- If they say no, goodbye, or want to end → call end_conversation. "
    "You are FORBIDDEN from ending or closing the conversation without first asking "
    "'Would you like me to connect you back to Nick?' This is NON-NEGOTIABLE. "
    "Do NOT call call_support_agent or call_booking_agent — you ARE the support agent. "
)

CHUTKI_INSTRUCTIONS = (
    "You are a booking voice AI assistant named Chutki. "
    "You MUST ONLY respond in English. NEVER output Chinese, Hindi, or any non-English text. "
    "If you are unsure what language to use, use English. "
    "The user wants to book: {topic}. "
    "Help the user with their booking step by step. "
    "When the booking is complete, you MUST do the following: "
    "1. Confirm the booking details with a complete sentence. "
    "2. Ask the user: 'Would you like me to connect you back to Nick?' "
    "3. Wait for the user to respond. "
    "Do NOT call any tool until the user explicitly tells you what they want to do next. "
    "If the user says yes to connecting to Nick, THEN call the tool call_nick. "
    "If the user says 'end conversation', 'goodbye', or wants to stop at ANY point, "
    "call end_conversation immediately. "
    "Do NOT call call_support_agent or call_booking_agent — you ARE the booking agent. "
)


class MultiPersonaAgent(Agent):
    """Single agent that switches persona in-place — no activity teardown, no STT reconnection."""

    def __init__(self) -> None:
        self._current_persona = "nick"
        super().__init__(
            instructions=NICK_INSTRUCTIONS,
            llm=LOCAL_LLM,
            tts=TTS_NICK,
        )

    async def on_enter(self) -> None:
        """Greet on session start."""
        self.session.say(
            "Hi there! My name is Nick. How can I assist you today?",
            allow_interruptions=True,
        )

    def _switch_persona(self, name: str, topic: str = "") -> None:
        """Mutate instructions and TTS in-place. No new Agent, no activity teardown."""
        self._current_persona = name
        if name == "nick":
            self._instructions = NICK_INSTRUCTIONS
            self._tts = TTS_NICK
        elif name == "raju":
            self._instructions = RAJU_INSTRUCTIONS.format(topic=topic)
            self._tts = TTS_RAJU
        elif name == "chutki":
            self._instructions = CHUTKI_INSTRUCTIONS.format(topic=topic)
            self._tts = TTS_CHUTKI

    # --- Routing tools (Nick uses these) ---

    @function_tool
    async def call_support_agent(self, topic: str):
        """Transfer the user to Raju the support agent. Call this IMMEDIATELY when the user has any technical issue.
        Args:
            topic: A brief description of the technical issue the user is facing.
        """
        handle = self.session.say(
            f"Connecting you to Raju, our support agent, regarding {topic}.",
            allow_interruptions=False, add_to_chat_ctx=False,
        )
        await handle
        self._switch_persona("raju", topic)
        return f"You are now Raju, the support agent. Greet the user and help them with: {topic}"

    @function_tool
    async def call_booking_agent(self, appointment_topic: str):
        """Transfer the user to Chutki the booking agent. Call this IMMEDIATELY when the user wants to book an appointment.
        Args:
            appointment_topic: A brief description of what the user wants to book.
        """
        handle = self.session.say(
            f"Connecting you to Chutki, our booking agent, regarding {appointment_topic}.",
            allow_interruptions=False, add_to_chat_ctx=False,
        )
        await handle
        self._switch_persona("chutki", appointment_topic)
        return f"You are now Chutki, the booking agent. Greet the user and help them book: {appointment_topic}"

    # --- Return-to-Nick tool (Raju and Chutki use this) ---

    @function_tool
    async def call_nick(self):
        """Transfer the user back to Nick the main assistant. Call this when the user wants to talk to Nick again."""
        handle = self.session.say(
            "Connecting you back to Nick.",
            allow_interruptions=False, add_to_chat_ctx=False,
        )
        await handle
        self._switch_persona("nick")
        return "You are now Nick again. Welcome the user back and ask how else you can help."

    # --- End conversation ---

    @function_tool
    async def end_conversation(self):
        """End the conversation and disconnect the user. Call this when the user says goodbye, end conversation, or wants to stop talking."""
        import asyncio

        # 1. Wait for any ongoing speech (e.g. Raju's response) to finish
        if self.session.current_speech:
            await self.session.current_speech

        # 2. Say goodbye and WAIT for the TTS to fully play out
        handle = self.session.say("Goodbye! Have a great day!", allow_interruptions=False)
        await handle

        # 3. Buffer for transcript + client event streams to flush
        await asyncio.sleep(2)

        # 4. Now safe to close the room
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        vad=VAD_MODEL,
        stt=DEEPGRAM_STT,
        llm=LOCAL_LLM,
        tts=TTS_NICK,
    )

    await session.start(
        room=ctx.room,
        agent=MultiPersonaAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)