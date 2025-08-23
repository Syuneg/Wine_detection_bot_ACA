import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define transforms
def get_transforms(augment=False):
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(base_transforms)

# Load model
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
        
        # Class names
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

# Telegram bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Start command received")
    await update.message.reply_text(
        "🍷 **Wine Classifier Bot**\n\n"
        "Send me a photo of wine and I'll classify it!\n"
        "I can identify 10 different wine types including red, white, and rosé varieties."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Photo received!")
    try:
        # Get the photo
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        print(f"Photo downloaded: {len(photo_bytes)} bytes")
        
        # Make prediction
        result = predict_wine(bytes(photo_bytes))
        
        if result:
            response = (
                f"🍷 **Wine Classification Result:**\n\n"
                f"**Predicted Type:** {result['predicted_class']}\n"
                f"**Confidence:** {result['confidence']:.2%}\n\n"
                f"**Top 3 Predictions:**\n"
            )
            
            # Get top 3 predictions
            all_probs = list(zip(CLASS_NAMES, result['all_confidences']))
            all_probs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (class_name, prob) in enumerate(all_probs[:3]):
                response += f"{i+1}. {class_name}: {prob:.2%}\n"
            
            predicted_class = result['predicted_class'].lower()
            if predicted_class == 'red':
                response += "\n🍇 **Category:** Red Wine"
            elif predicted_class == 'white':
                response += "\n🥂 **Category:** White Wine"  
            elif predicted_class == 'rose':
                response += "\n🌹 **Category:** Rosé Wine"
            elif predicted_class == 'sparkling' or predicted_class == 'champagne':
                response += "\n✨ **Category:** Sparkling Wine"
            elif predicted_class == 'dessert' or predicted_class == 'icewine':
                response += "\n🍰 **Category:** Dessert Wine"
            elif predicted_class == 'fortified':
                response += "\n⚡ **Category:** Fortified Wine"
            elif predicted_class == 'sake_rice_wine':
                response += "\n🍶 **Category:** Sake/Rice Wine"
            else:
                response += "\n🎯 **Category:** Specialty Wine"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            print(f"Prediction sent: {result['predicted_class']}")
            
    except Exception as e:
        print(f"Error in handle_photo: {e}")
        await update.message.reply_text("❌ Error processing image. Please try another photo.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Error: {context.error}")
    if update:
        await update.message.reply_text("❌ Something went wrong!")

def main():
    TOKEN = "8357680328:AAED3VVJ84kIrJAP2hhbm7JcT_UWaghA5js"
    
    if TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ ERROR: Please add Telegram bot token!")
        return
    
    print("Creating application...")
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
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