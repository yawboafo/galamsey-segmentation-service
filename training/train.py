import os
import torch
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation, 
    TrainingArguments, 
    Trainer
)
from training.dataset import GalamseyDataset, get_train_transform, get_val_transform
from training.metrics import compute_metrics
from training.config import train_settings
import argparse

def train():
    # 1. Initialize Processor and Model
    checkpoint = train_settings.MODEL_CHECKPOINT
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    processor.do_reduce_labels = False # We handle masks ourselves
    # ID to Label mapping
    id2label = {
        0: "background", 
        1: "galamsey_pit", 
        2: "vegetation_loss", 
        3: "road", 
        4: "water_turbid"
    }
    label2id = {v: k for k, v in id2label.items()}
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # Important for fine-tuning
    )
    
    # 2. Prepare Datasets
    train_dataset = GalamseyDataset(
        image_dir=os.path.join(train_settings.TRAIN_DIR, "images"),
        mask_dir=os.path.join(train_settings.TRAIN_DIR, "masks"),
        processor=processor,
        image_size=train_settings.IMAGE_SIZE,
        transform=get_train_transform(train_settings.IMAGE_SIZE)
    )
    
    val_dataset = GalamseyDataset(
        image_dir=os.path.join(train_settings.VAL_DIR, "images"),
        mask_dir=os.path.join(train_settings.VAL_DIR, "masks"),
        processor=processor,
        image_size=train_settings.IMAGE_SIZE,
        transform=get_val_transform(train_settings.IMAGE_SIZE)
    )
    
    # 3. Training Arguments
    args = TrainingArguments(
        output_dir=train_settings.OUTPUT_DIR,
        learning_rate=train_settings.LEARNING_RATE,
        num_train_epochs=train_settings.NUM_EPOCHS,
        per_device_train_batch_size=train_settings.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=train_settings.VAL_BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False, # Standard for HF segmentation trainer
        push_to_hub=train_settings.PUSH_TO_HUB,
        hub_model_id=train_settings.HF_MODEL_REPO,
        hub_token=train_settings.HF_TOKEN,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
    )
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 5. Run Training
    print("Starting training...")
    trainer.train()
    
    # 6. Save Best Model
    print(f"Saving best model to {train_settings.BEST_MODEL_DIR}")
    trainer.save_model(train_settings.BEST_MODEL_DIR)
    processor.save_pretrained(train_settings.BEST_MODEL_DIR)
    
    if train_settings.PUSH_TO_HUB:
        trainer.push_to_hub()

if __name__ == "__main__":
    train()
