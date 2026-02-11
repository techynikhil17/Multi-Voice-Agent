from dotenv import load_dotenv
import os
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.agents.llm import function_tool, ChatContext
from livekit.plugins import (
    openai,
    elevenlabs,
    deepgram,
    silero,
    noise_cancellation
)

from generic_agent import GenericAgent

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


class StarterAgent(GenericAgent):
    def __init__(self, chat_ctx: ChatContext = None) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant named Nick. "
                "You MUST ONLY respond in English. NEVER output Chinese, Hindi, or any non-English text. "
                "If you are unsure what language to use, use English. "
                "Your role is to route users to the right agent. "
                "If the user mentions ANY technical issue, problem, or needs help fixing something, "
                "IMMEDIATELY call the tool call_support_agent. Do NOT troubleshoot or discuss the issue yourself. "
                "If the user mentions wanting to book, schedule, or make an appointment, "
                "IMMEDIATELY call the tool call_booking_agent. Do NOT handle the booking yourself. "
                "If the user wants to end the conversation, says goodbye, or says 'end conversation', "
                "call the tool end_conversation to disconnect. "
                "If the user just says hello or asks a general question, respond normally. "
                "Do NOT treat greetings like 'hello', 'hi', or 'hey' as technical issues. "
            ),
            llm=LOCAL_LLM,
            tts=TTS_NICK,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        """Deterministic greeting based on user history."""
        has_user_history = any(m.role == "user" for m in self.chat_ctx.messages())
        if has_user_history:
            self.session.say(
                "Hi again, welcome back! Is there anything else I can help you with?",
                allow_interruptions=True
            )
        else:
            self.session.say(
                "Hi there! My name is Nick. How can I assist you today?",
                allow_interruptions=True
            )

    @function_tool
    async def call_support_agent(self, topic: str):
        """Transfer the user to Raju the support agent. Call this IMMEDIATELY when the user has any technical issue.
        Args:
            topic: A brief description of the technical issue the user is facing.
        """
        handle = self.session.say(f"Connecting you to Raju, our support agent, regarding {topic}.", allow_interruptions=False, add_to_chat_ctx=False)
        await handle
        return SupportAgent(topic=topic, chat_ctx=self.chat_ctx)

    @function_tool
    async def call_booking_agent(self, appointment_topic: str):
        """Transfer the user to Chutki the booking agent. Call this IMMEDIATELY when the user wants to book an appointment.
        Args:
            appointment_topic: A brief description of what the user wants to book.
        """
        handle = self.session.say(f"Connecting you to Chutki, our booking agent, regarding {appointment_topic}.", allow_interruptions=False, add_to_chat_ctx=False)
        await handle
        return BookingAgent(appointment_topic=appointment_topic, chat_ctx=self.chat_ctx)


class SupportAgent(GenericAgent):
    def __init__(self, topic: str, chat_ctx: ChatContext = None) -> None:
        self._topic = topic
        super().__init__(
            instructions=(
                "You are a support voice AI assistant named Raju. "
                "You MUST ONLY respond in English. NEVER output Chinese, Hindi, or any non-English text. "
                "If you are unsure what language to use, use English. "
                f"The user is facing an issue with: {topic}. "
                "Help the user resolve their technical issue step by step. "
                "When the user says the issue is resolved, you MUST do the following: "
                "1. Acknowledge that the issue is resolved with a complete sentence. "
                "2. Ask the user: 'Would you like me to connect you back to Nick?' "
                "3. Wait for the user to respond. "
                "Do NOT call any tool until the user explicitly tells you what they want to do next. "
                "If the user says yes to connecting to Nick, THEN call the tool call_nick. "
                "If the user says 'end conversation', 'goodbye', or wants to stop at ANY point, "
                "call end_conversation immediately."
            ),
            llm=LOCAL_LLM,
            tts=TTS_RAJU,
            chat_ctx=chat_ctx,
        )
        self._greeting = f"Hi, I am Raju, your support agent. I understand you are facing an issue with {topic}. Let me help you with that."

    @function_tool
    async def call_nick(self):
        """Transfer the user back to Nick the main assistant. Call this when the user wants to talk to Nick again."""
        handle = self.session.say("Connecting you back to Nick.", allow_interruptions=False, add_to_chat_ctx=False)
        await handle
        return StarterAgent(chat_ctx=self.chat_ctx)


class BookingAgent(GenericAgent):
    def __init__(self, appointment_topic: str, chat_ctx: ChatContext = None) -> None:
        self._appointment_topic = appointment_topic
        super().__init__(
            instructions=(
                "You are a booking voice AI assistant named Chutki. "
                "You MUST ONLY respond in English. NEVER output Chinese, Hindi, or any non-English text. "
                "If you are unsure what language to use, use English. "
                f"The user wants to book: {appointment_topic}. "
                "Help the user with their booking step by step. "
                "When the booking is complete, you MUST do the following: "
                "1. Confirm the booking details with a complete sentence. "
                "2. Ask the user: 'Would you like me to connect you back to Nick?' "
                "3. Wait for the user to respond. "
                "Do NOT call any tool until the user explicitly tells you what they want to do next. "
                "If the user says yes to connecting to Nick, THEN call the tool call_nick. "
                "If the user says 'end conversation', 'goodbye', or wants to stop at ANY point, "
                "call end_conversation immediately."
            ),
            llm=LOCAL_LLM,
            tts=TTS_CHUTKI,
            chat_ctx=chat_ctx,
        )
        self._greeting = f"Hi, I am Chutki, your booking agent. I understand you want to book {appointment_topic}. Let me help you with that."

    @function_tool
    async def call_nick(self):
        """Transfer the user back to Nick the main assistant. Call this when the user wants to talk to Nick again."""
        handle = self.session.say("Connecting you back to Nick.", allow_interruptions=False, add_to_chat_ctx=False)
        await handle
        return StarterAgent(chat_ctx=self.chat_ctx)


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
        agent=StarterAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVC(),
            ),
        ),
    )

if __name__ == "__main__":
    agents.cli.run_app(server)