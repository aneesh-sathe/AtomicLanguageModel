import os

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


# Train and save tokenizer
def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files="dataset/marathi_pretrain.txt",
        vocab_size=32000,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
            "### सूचना:",
            "### उत्तर:",
            "### इनपुट:",
        ],
    )

    os.makedirs("marathi_tokenizer", exist_ok=True)
    tokenizer.save_model("marathi_tokenizer")

    # Save tokenizer config manually
    with open("marathi_tokenizer/special_tokens_map.json", "w", encoding="utf-8") as f:
        f.write("""{
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "additional_special_tokens": ["### सूचना:", "### उत्तर:", "### इनपुट:"]
        }""")

    with open("marathi_tokenizer/tokenizer_config.json", "w", encoding="utf-8") as f:
        f.write("""{
            "add_prefix_space": true,
            "model_max_length": 512,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["### सूचना:", "### उत्तर:", "### इनपुट:"]
        }""")


train_tokenizer()
tokenizer = GPT2TokenizerFast.from_pretrained("marathi_tokenizer")
print(tokenizer.pad_token)


print(
    len(
        tokenizer.encode(
            "वारशानुसार राष्ट्रीय सूचीबद्ध इमारत म्हणून नियुक्त आहे की नाही हे दर्शवते. काही प्रकरणांमध्ये, अतिरिक्त माहिती तिरस्करणीत दिली जाते. ही यादी हेरिटेज गेटवे संकेतस्थळाद्वारे उपलब्ध असलेल्या ऐतिहासिक पर्यावरणीय नोंदींच्या शोधावर आधारित आहे."
        )
    )
)
