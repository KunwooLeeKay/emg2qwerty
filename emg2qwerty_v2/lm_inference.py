from __future__ import annotations

import abc

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field, InitVar
from typing import Any, ClassVar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import kenlm
import numpy as np

from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.data import LabelData
import openai

def gpt_autocorrect(sentence: str, model="gpt-4o-mini") -> str:
    client = openai.OpenAI(api_key = "API-KEY")
    prompt = (
        "Correct the spelling *if needed* and return only the corrected word.\n"
        "Do not include punctuation or explanation.\n"
        "Do not change the case of the word.\n\n"
        f"{sentence.strip()}"
    )
    print(f"[DEBUG] Sending prompt to GPT: '{sentence.strip()}'")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"[DEBUG] GPT response: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        return sentence


@dataclass
class CTCBeamDecoderWithGPTWord(CTCBeamDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        words = clean_text.split()
        corrected_words = []
        for word in words:
            corrected_words.append(gpt_autocorrect(word))
        label_data.text = ' '.join(corrected_words)
        return label_data

@dataclass
class CTCBeamDecoderWithGPTSentence(CTCBeamDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        corrected = gpt_autocorrect(clean_text)
        label_data.text = corrected
        return label_data

@dataclass
class CTCGreedyDecoderWithGPTWord(CTCGreedyDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        words = clean_text.split()
        corrected_words = []
        for word in words:
            corrected_words.append(gpt_autocorrect(word))
        label_data.text = ' '.join(corrected_words)
        return label_data
@dataclass
class CTCGreedyDecoderWithGPTSentence(CTCGreedyDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        corrected = gpt_autocorrect(clean_text)
        label_data.text = corrected
        return label_data
