import logging

# 1. Configure the root logger and silence noisy loggers BEFORE any other imports.
# This is the most aggressive way to prevent warnings during the import phase.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

# 2. Now import the rest of the dependencies.
import asyncio
import sqlite3
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

from config import TELEGRAM_BOT_TOKEN
from graph import build_graph
from streaming import extract_text

# Setup local logger for this module
logger = logging.getLogger(__name__)

# Initialize the LangGraph agent
app = build_graph()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the user types /start."""
    await update.message.reply_text(
        "Hello! I am your AI Agent. You can chat with me, ask me to read files, "
        "execute commands, or even make web requests!\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/clear - Reset your conversation history"
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history for the current user."""
    user_id = str(update.effective_user.id)
    
    try:
        # Connect to the same database used by the agent
        conn = sqlite3.connect(".data/history.db")
        cursor = conn.cursor()
        
        # Delete all checkpoints and writes for this thread_id
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (user_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        
        await update.message.reply_text("Your conversation history has been cleared! You can start a new chat now.")
    except Exception as e:
        logger.error(f"Error clearing history for user {user_id}: {e}")
        await update.message.reply_text("Failed to clear conversation history. Please try again later.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages from Telegram users."""
    user_id = update.effective_user.id
    user_text = update.message.text

    if not user_text:
        return

    # Use user_id as thread_id for conversation persistence
    config = {"configurable": {"thread_id": str(user_id)}}

    # Send "typing..." action to improve UX
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)

    try:
        # Run the agent in a separate thread to avoid blocking the event loop
        def run_agent():
            full_response = ""
            # We iterate through the stream to get the final content
            # Note: app.stream returns chunks of messages
            for chunk, metadata in app.stream(
                {"messages": [{"role": "user", "content": user_text}]},
                config=config,
                stream_mode="messages",
            ):
                # We only care about the agent's response chunks
                if metadata.get("langgraph_node") == "agent":
                    text = extract_text(chunk.content)
                    if text:
                        full_response += text
            return full_response

        response_text = await asyncio.to_thread(run_agent)

        if response_text:
            await update.message.reply_text(response_text)
        else:
            await update.message.reply_text("I'm sorry, I couldn't generate a response.")

    except Exception as e:
        logger.error(f"Error processing message from user {user_id}: {e}")
        await update.message.reply_text("An error occurred while processing your request. Please try again later.")

def main() -> None:
    """Starts the Telegram bot."""
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("clear", clear_command))

    # Register the message handler
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(message_handler)

    logger.info("Telegram bot is starting...")
    # run_polling is a blocking call that manages its own event loop
    application.run_polling()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped.")
