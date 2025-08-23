import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import numpy as np
import logging
import random

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_transforms(augment=False):
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(base_transforms)

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        print("Checkpoint loaded successfully")
        
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 10)
        
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        print("✅ ResNet50 model loaded successfully!")
        print(f"Model has {num_ftrs} features and 10 output classes")
        
        class_names = [
            'Red', 'White', 'Rose', 'Specialty', 'Sparkling',
            'Sake_Rice_wine', 'Icewine', 'Fortified', 'Dessert', 'Champagne'
        ]
        
        return model, device, class_names
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise e
    
try:
    model, device, CLASS_NAMES = load_model('best_model.pth')
    transform = get_transforms(augment=False)
    print(f"Model ready! Classes: {CLASS_NAMES}")
except Exception as e:
    print(f"Failed to initialize model: {e}")
    exit()

def predict_wine(image):
    try:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            'predicted_class': CLASS_NAMES[predicted.item()],
            'confidence': confidence.item(),
            'all_confidences': probabilities.cpu().numpy()[0],
            'predicted_idx': predicted.item()
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Welcome messages
WELCOME_MESSAGES = [
    "Hello! I'm your wine expert bot 🍷",
    "Hi there! I can help identify wine types from photos",
    "Greetings! Send me a wine photo and I'll tell you what type it is",
    "Welcome! I'm a wine classification bot ready to analyze your photos"
]

# Response messages
RESPONSE_INTROS = [
    "Let me analyze this wine...",
    "Examining your wine photo...",
    "Taking a closer look at this...",
    "Analyzing the color and characteristics..."
]

# Telegram bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Start command received")
    welcome = random.choice(WELCOME_MESSAGES)
    await update.message.reply_text(
        f"{welcome}\n\n"
        "🍷 **Wine Classifier Bot**\n\n"
        "Simply send me a clear photo of wine and I'll classify it for you!\n"
        "I can identify 10 different wine types including:\n"
        "• Red, White, and Rosé wines\n"
        "• Sparkling wines and Champagne\n"
        "• Dessert and Fortified wines\n"
        "• Sake/Rice wine and more!\n\n"
        "Just send a photo to get started! 📸"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 **How to use the Wine Classifier Bot:**\n\n"
        "1. Take a clear photo of the wine you want to identify\n"
        "2. Make sure the wine is well-lit and visible\n"
        "3. Send the photo to me\n"
        "4. I'll analyze it and tell you what type of wine it is!\n\n"
        "I can identify 10 different wine types with my AI model.\n\n"
        "Just send a photo whenever you're ready!"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.lower()
    
    greetings = ['hi', 'hello', 'hey', 'hola', 'greetings']
    thanks = ['thanks', 'thank you', 'thx', 'ty']
    
    if any(word in user_message for word in greetings):
        await update.message.reply_text(
            f"{random.choice(WELCOME_MESSAGES)} Send me a wine photo to get started! 📸"
        )
    elif any(word in user_message for word in thanks):
        await update.message.reply_text(
            "You're welcome! 🍷 Feel free to send more wine photos anytime!"
        )
    elif 'what can you do' in user_message or 'how do you work' in user_message:
        await help_command(update, context)
    else:
        await update.message.reply_text(
            "I'm a wine classification bot! 🍷\n\n"
            "Send me a photo of wine and I'll tell you what type it is. "
            "Or use /help to see how I work!"
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Photo received!")
    try:
        processing_msg = await update.message.reply_text(
            f"{random.choice(RESPONSE_INTROS)} 🔍"
        )
        
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        print(f"Photo downloaded: {len(photo_bytes)} bytes")
        
        result = predict_wine(bytes(photo_bytes))
        
        if result:
            await context.bot.delete_message(
                chat_id=update.message.chat_id,
                message_id=processing_msg.message_id
            )
            
            response = (
                f"🍷 **Wine Classification Result:**\n\n"
                f"**Predicted Type:** {result['predicted_class']}\n"
                f"**Confidence:** {result['confidence']:.2%}\n\n"
                f"**Top 3 Predictions:**\n"
            )
            
            all_probs = list(zip(CLASS_NAMES, result['all_confidences']))
            all_probs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (class_name, prob) in enumerate(all_probs[:3]):
                response += f"{i+1}. {class_name}: {prob:.2%}\n"
            
            predicted_class = result['predicted_class'].lower()
            if predicted_class == 'red':
                response += "\n🍇 **Red Wine** - Known for bold flavors and darker color from extended grape skin contact during fermentation."
            elif predicted_class == 'white':
                response += "\n🥂 **White Wine** - Typically lighter, crisper, and made from green or yellow-colored grapes."  
            elif predicted_class == 'rose':
                response += "\n🌹 **Rosé Wine** - Gets its pink color from limited skin contact with red grapes, offering a middle ground between red and white."
            elif predicted_class == 'sparkling':
                response += "\n✨ **Sparkling Wine** - Contains significant levels of carbon dioxide, making it fizzy."
            elif predicted_class == 'champagne':
                response += "\n🍾 **Champagne** - A specific type of sparkling wine from the Champagne region of France."
            elif predicted_class == 'dessert':
                response += "\n🍰 **Dessert Wine** - Sweet wines typically served with dessert."
            elif predicted_class == 'icewine':
                response += "\n❄️ **Icewine** - A type of dessert wine produced from grapes that have been frozen while still on the vine."
            elif predicted_class == 'fortified':
                response += "\n⚡ **Fortified Wine** - Wine to which a distilled spirit (like brandy) has been added."
            elif predicted_class == 'sake_rice_wine':
                response += "\n🍶 **Sake/Rice Wine** - A Japanese alcoholic beverage made by fermenting rice."
            else:
                response += "\n🎯 **Specialty Wine** - A unique or specialty wine variety."
            
            await update.message.reply_text(response, parse_mode='Markdown')
            print(f"Prediction sent: {result['predicted_class']}")
            
    except Exception as e:
        print(f"Error in handle_photo: {e}")
        await update.message.reply_text(
            "❌ Sorry, I had trouble processing that image.\n\n"
            "Please try again with a clearer photo of the wine, preferably in good lighting. 📸"
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Error: {context.error}")
    if update:
        await update.message.reply_text(
            "❌ Something went wrong! Please try again or send a different photo."
        )

def main():
    TOKEN = "8357680328:AAED3VVJ84kIrJAP2hhbm7JcT_UWaghA5js"
    
    if TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ ERROR: Please add Telegram bot token!")
        return
    
    print("Creating application...")
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_error_handler(error_handler)
    
    print("=" * 50)
    print("🍷 Wine Classifier Bot Starting...")
    print(f"Model: ResNet50 with {len(CLASS_NAMES)} classes")
    print("Classes:", CLASS_NAMES)
    print("=" * 50)
    
    try:
        application.run_polling()
    except Exception as e:
        print(f"Failed to start bot: {e}")

if __name__ == "__main__":
    main()