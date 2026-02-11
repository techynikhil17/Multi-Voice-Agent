from livekit.agents import Agent
from livekit.agents.llm import function_tool
from livekit.agents.job import get_job_context
from livekit import api


class GenericAgent(Agent):
    # Subclasses should set this to their greeting text
    _greeting: str = ""

    async def on_enter(self) -> None:
        """Speak the agent's greeting using direct TTS (no LLM, deterministic)."""
        if self._greeting:
            self.session.say(self._greeting, allow_interruptions=True)

    @function_tool
    async def end_conversation(self):
        """End the conversation and disconnect the user. Call this when the user says goodbye, end conversation, or wants to stop talking."""
        self.session.say("Goodbye! Have a great day!", allow_interruptions=False)

        import asyncio
        await asyncio.sleep(3)

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )