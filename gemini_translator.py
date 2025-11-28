"""
Gemini API translator module with glossary support
Uses REST API to avoid dependency conflicts
Supports rotating API keys to handle rate limits
"""

import json
import time
import requests


class GeminiTranslator:
    def __init__(self, api_key, logger):
        self.logger = logger
        self.glossary = {}
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # Support multiple API keys (comma-separated or single)
        if api_key:
            # Parse multiple keys if provided as comma-separated
            self.api_keys = [k.strip() for k in api_key.split(',') if k.strip()]
            self.api_key = self.api_keys[0] if self.api_keys else None
            self.current_key_index = 0
        else:
            self.api_keys = []
            self.api_key = None
        
        if not self.api_keys:
            self.logger("WARNING: No Gemini API key provided")
            return
        
        self.logger(f"✓ Gemini REST API client initialized with {len(self.api_keys)} API key(s)")
    
    def _get_current_key(self):
        """Get the current API key"""
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]
    
    def _rotate_key(self):
        """Rotate to the next API key"""
        if len(self.api_keys) <= 1:
            return False
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        self.logger(f"  🔄 Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        return self.current_key_index != old_index
    
    def _call_gemini_api(self, model, prompt, temperature=0.3):
        """Call Gemini REST API with automatic key rotation on 429 errors"""
        if not self.api_keys:
            raise Exception("No API keys available")
        
        # Try each key before giving up
        keys_tried = 0
        last_error = None
        
        while keys_tried < len(self.api_keys):
            current_key = self._get_current_key()
            url = f"{self.base_url}/{model}:generateContent?key={current_key}"
            
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": temperature
                }
            }
            
            try:
                response = requests.post(url, json=payload, timeout=120)
                
                # Check for rate limit (429) - rotate key
                if response.status_code == 429:
                    self.logger(f"  ⚠ API key {self.current_key_index + 1} hit rate limit (429)")
                    keys_tried += 1
                    if self._rotate_key():
                        continue  # Try next key
                    else:
                        # Only one key or all keys exhausted
                        response.raise_for_status()
                
                response.raise_for_status()
                
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        return candidate['content']['parts'][0]['text']
                
                raise Exception("No valid response from Gemini API")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    last_error = e
                    keys_tried += 1
                    if self._rotate_key():
                        continue
                raise
            except Exception as e:
                raise
        
        # All keys exhausted
        if last_error:
            raise last_error
        raise Exception("All API keys exhausted")
    
    def translate_description(self, description_html, raw_novel_name=None, source_lang='zh-CN', target_lang='en'):
        """
        Translate novel description with prompt to clean and extract only the description
        """
        if not self.api_key:
            self.logger("WARNING: Gemini API key not available for description translation")
            return description_html
        
        prompt = f"""You are a professional translator specializing in Chinese web novels.

Task: Translate the following Chinese novel description to English. 

Important instructions:
1. ONLY return the main story synopsis/description
2. Remove ALL markdown formatting (**, ##, bullets, etc.) - plain text only
3. Remove character profiles, tags, reading guides, author notes, upcoming novels
4. Remove "Latest Chapter:", "Update:", footers, advertisements
5. Remove translator notes, character descriptions, themes
6. Keep ONLY the core story plot description
7. Natural, engaging translation with proper paragraph breaks
8. No explanations, no comments - ONLY the synopsis text

Chinese text to translate:
{description_html}

English translation:"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                translated = self._call_gemini_api('gemini-flash-latest', prompt, temperature=0.3).strip()
                
                # Verify translation actually happened (not just returned original)
                if translated == description_html or len(translated) < 10:
                    raise Exception("Translation returned original or empty text")
                
                # Add raw novel name at the end if provided
                if raw_novel_name:
                    translated = f"{translated}\n\nRaw Novel Name: {raw_novel_name}"
                
                self.logger("  ✓ Description translated with Gemini")
                return translated
                
            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_retries - 1:
                    self.logger(f"  ⚠ Gemini description attempt {attempt + 1} failed: {e}")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger(f"  ✗ Gemini description translation failed after {max_retries} attempts: {e}")
                    # CRITICAL: Return None to signal translation failure
                    return None
    
    def generate_glossary(self, chapters_data, max_chapters=None):
        """
        Generate a glossary of names and terms from all chapters
        MANDATORY - returns None on failure to abort the process
        
        Args:
            chapters_data: List of dicts with 'title' and 'content' (Chinese)
            max_chapters: Not used (kept for compatibility)
        
        Returns:
            dict: Glossary mapping Chinese terms to English translations
            None: If generation fails (signals abort)
        """
        if not self.api_key:
            self.logger("WARNING: Gemini API key not available for glossary generation")
            return None
        
        self.logger(f"\n{'='*50}")
        self.logger("Generating Translation Glossary (MANDATORY)")
        self.logger(f"{'='*50}")
        self.logger(f"Analyzing all {len(chapters_data)} chapters...")
        
        # Batch processing configuration
        BATCH_SIZE = 30
        self.glossary = {}
        
        # Split chapters into batches
        batches = [chapters_data[i:i + BATCH_SIZE] for i in range(0, len(chapters_data), BATCH_SIZE)]
        self.logger(f"  Split into {len(batches)} batches of {BATCH_SIZE} chapters each")
        
        for batch_idx, batch in enumerate(batches):
            self.logger(f"\n  Processing Batch {batch_idx + 1}/{len(batches)} ({len(batch)} chapters)...")
            
            # Combine chapters in this batch
            combined_text = "\n\n".join([
                f"Chapter {ch['idx']}:\n{ch['content']}"
                for ch in batch
            ])
            
            # Include existing glossary to maintain consistency
            existing_glossary_context = ""
            if self.glossary:
                glossary_entries = "\n".join([f"- {cn} = {en}" for cn, en in list(self.glossary.items())[:100]])
                existing_glossary_context = f"""
Existing Glossary (MAINTAIN CONSISTENCY WITH THESE):
{glossary_entries}
"""
            
            prompt = f"""You are a professional translator for Chinese web novels.

Task: Analyze the following Chinese novel chapters and create/update a consistent English glossary.

Instructions:
1. Extract ALL important names (characters, places, organizations)
2. Extract ALL cultivation terms, skill names, and special terminology
3. Provide consistent English translations
4. IF a term is already in the "Existing Glossary", USE THE SAME TRANSLATION
5. Return ONLY a JSON object in this exact format:

{{
  "characters": {{"中文名": "English Name", ...}},
  "places": {{"中文地名": "English Place", ...}},
  "terms": {{"中文术语": "English Term", ...}}
}}

{existing_glossary_context}

Chinese chapters to analyze:
{combined_text}

JSON glossary:"""
            
            # Retry logic for this batch
            wait_times = [60, 120, 240, 480]
            max_retries = len(wait_times)
            batch_success = False
            
            for attempt in range(max_retries):
                try:
                    self.logger(f"    Batch {batch_idx + 1} attempt {attempt + 1}...")
                    
                    response_text = self._call_gemini_api('gemini-flash-latest', prompt, temperature=0.2).strip()
                    
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                    
                    batch_glossary = json.loads(response_text)
                    
                    # Merge into main glossary
                    new_entries = 0
                    for category in ['characters', 'places', 'terms']:
                        if category in batch_glossary:
                            for cn, en in batch_glossary[category].items():
                                if cn not in self.glossary:
                                    self.glossary[cn] = en
                                    new_entries += 1
                                # If exists, keep existing translation for consistency
                    
                    self.logger(f"    ✓ Batch processed: Added {new_entries} new entries")
                    batch_success = True
                    break
                    
                except Exception as e:
                    self.logger(f"    ✗ Batch attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(wait_times[attempt])
            
            if not batch_success:
                self.logger(f"  ✗ CRITICAL: Batch {batch_idx + 1} failed after retries")
                self.logger(f"  Aborting glossary generation")
                return None
            
            # Small delay between batches to be nice to API
            if batch_idx < len(batches) - 1:
                time.sleep(5)
        
        self.logger(f"\n✓ Final Glossary generated with {len(self.glossary)} entries")
        return self.glossary
    
    def translate_chapter_content(self, content, chapter_number, glossary=None, google_translator=None):
        """
        Translate chapter content using Gemini with glossary consistency
        Enhanced with multi-level fallback strategy
        
        Args:
            content: Chinese chapter content
            chapter_number: Chapter number for context
            glossary: Optional glossary dict (uses self.glossary if not provided)
            google_translator: Google Translate instance for fallback
        
        Returns:
            tuple: (translated_content, translation_method)
            translation_method: 'gemini', 'gemini_censored', or 'google'
        """
        if not self.api_key:
            self.logger("    WARNING: Gemini API key not available")
            return None, 'failed'
        
        if glossary is None:
            glossary = self.glossary
        
        # Build glossary instructions
        glossary_text = ""
        if glossary:
            glossary_entries = "\n".join([f"- {cn} = {en}" for cn, en in list(glossary.items())[:50]])
            glossary_text = f"""
Use this glossary for consistent translations:
{glossary_entries}
"""
        
        prompt = f"""You are a professional translator for Chinese web novels.

Task: Translate the following Chinese chapter to natural, fluent English.

Instructions:
1. Maintain narrative flow and readability
2. Use the provided glossary for consistency with previous chapters
3. Keep the same paragraph structure
4. Translate cultivation terms naturally
5. Remove ALL markdown formatting (**, ##, etc.) - plain text only
6. Do NOT include any notes, explanations, or meta-commentary
7. Output ONLY the translated chapter content

{glossary_text}

Chapter {chapter_number} content (Chinese):
{content}

English translation:"""
        
        # ATTEMPT 1: Try Gemini with original content (with retries for API errors)
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                translated = self._call_gemini_api('gemini-flash-latest', prompt, temperature=0.3).strip()
                return translated, 'gemini'
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check for safety/content filtering errors - DON'T RETRY, go to fallback
                if 'SAFETY' in error_msg.upper() or 'BLOCK' in error_msg.upper() or 'HARM' in error_msg.upper():
                    self.logger(f"    ⚠ Gemini safety filter triggered: {error_msg}")
                    break  # Exit retry loop, go to fallback strategy
                
                # For other errors (API, rate limit, etc.), retry
                if attempt < max_retries - 1:
                    self.logger(f"    ⚠ Gemini attempt {attempt + 1} failed: {error_msg}")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    self.logger(f"    ✗ Gemini translation failed after {max_retries} attempts: {error_msg}")
                    return None, 'failed'
        
        # Only reach here if safety filter triggered
        if last_error:
            error_msg = str(last_error)
            if 'SAFETY' in error_msg.upper() or 'BLOCK' in error_msg.upper() or 'HARM' in error_msg.upper():
                self.logger(f"    ⚠ Gemini safety filter triggered")
                
                # ATTEMPT 2: Translate with Google Translate, then censor and retry Gemini
                if google_translator:
                    self.logger(f"    → Attempting Google Translate + censoring + Gemini retry...")
                    
                    try:
                        # Translate with Google first
                        google_translation = google_translator.translate(content)
                        
                        # Censor potentially problematic words
                        censored_translation = self._censor_content(google_translation)
                        
                        # Retry Gemini with censored English text
                        retry_prompt = f"""You are a professional editor for web novels.

Task: Improve and polish the following English text while maintaining consistency with the glossary.

Instructions:
1. Fix any awkward phrasing or grammar
2. Use the glossary for character/place names
3. Keep the same paragraph structure
4. Make the text flow naturally
5. Output ONLY the polished content

{glossary_text}

Text to polish:
{censored_translation}

Polished version:"""
                        
                        polished = self._call_gemini_api('gemini-flash-latest', retry_prompt, temperature=0.3).strip()
                        self.logger(f"    ✓ Gemini succeeded with censored content")
                        return polished, 'gemini_censored'
                        
                    except Exception as retry_error:
                        self.logger(f"    ✗ Gemini retry also failed: {retry_error}")
                        # ATTEMPT 3: Fall back to Google Translate only
                        self.logger(f"    → Using Google Translate as final fallback")
                        return google_translation, 'google'
                else:
                    return None, 'failed'
        else:
            # No error or non-safety error after retries
            return None, 'failed'
    
    def _censor_content(self, text):
        """
        Censor potentially problematic words for content filtering
        
        Args:
            text: Text to censor
            
        Returns:
            Censored text
        """
        # Common words that might trigger filters (especially for BL/gay novels)
        censor_map = {
            # Explicit content
            'dick': 'member',
            'cock': 'member',
            'penis': 'member',
            'pussy': 'flower',
            'vagina': 'flower',
            'ass': 'behind',
            'asshole': 'behind',
            'anal': 'intimate',
            'sex': 'intimacy',
            'sexy': 'attractive',
            'sexual': 'intimate',
            'fuck': 'embrace',
            'fucking': 'embracing',
            'fucked': 'embraced',
            'cum': 'finish',
            'cumming': 'finishing',
            'orgasm': 'peak',
            'aroused': 'excited',
            'erection': 'reaction',
            'hard-on': 'reaction',
            'masturbate': 'touch',
            'penetrate': 'enter',
            'penetration': 'entry',
            'thrust': 'move',
            'thrusting': 'moving',
            'moan': 'sound',
            'moaning': 'sounding',
            'groan': 'sound',
            'groaning': 'sounding',
            'lust': 'desire',
            'lustful': 'desirous',
            'seduce': 'attract',
            'seduction': 'attraction',
            'naked': 'unclothed',
            'nude': 'bare',
            'breast': 'chest',
            'nipple': 'tip',
            'kiss': 'touch',
            'kissing': 'touching',
            'lick': 'taste',
            'licking': 'tasting',
            'suck': 'draw',
            'sucking': 'drawing',
            # Violence (keep minimal since you said it's not about violence)
            'blood': 'energy',
            'bloody': 'intense',
            'corpse': 'body',
            'tortured': 'pressured',
            'pain': 'discomfort',
            'painful': 'difficult',
            'weapon': 'tool',
            'sword': 'blade',
            'knife': 'blade',
            'attack': 'strike',
            'attacked': 'struck',
            'violent': 'intense',
            'violence': 'intensity',
        }
        
        censored = text
        for original, replacement in censor_map.items():
            # Case-insensitive replacement, preserve capitalization
            import re
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            
            def replace_func(match):
                orig = match.group(0)
                if orig[0].isupper():
                    return replacement.capitalize()
                return replacement
            
            censored = pattern.sub(replace_func, censored)
        
        return censored
    
    def save_glossary(self, novel_id, file_manager):
        """Save glossary to file for future reference"""
        if not self.glossary:
            return
        
        try:
            import os
            novel_dir = os.path.join('novels', f'novel_{novel_id}')
            os.makedirs(novel_dir, exist_ok=True)
            
            filepath = os.path.join(novel_dir, 'glossary.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.glossary, f, ensure_ascii=False, indent=2)
            
            self.logger(f"✓ Glossary saved to {filepath}")
        except Exception as e:
            self.logger(f"✗ Failed to save glossary: {e}")
    
    def load_glossary(self, novel_id):
        """Load existing glossary from file"""
        try:
            import os
            filepath = os.path.join('novels', f'novel_{novel_id}', 'glossary.json')
            
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.glossary = json.load(f)
                self.logger(f"✓ Loaded existing glossary with {len(self.glossary)} entries")
                return True
            return False
        except Exception as e:
            self.logger(f"✗ Failed to load glossary: {e}")
            return False
