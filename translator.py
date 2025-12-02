"""
Translation module using deep-translator (free Google Translate API)
This is used for titles and as fallback for content translation
"""

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False


class Translator:
    def __init__(self, project_id, logger, credentials_file=None):
        self.logger = logger
        self.client = None
        self.service = None
        
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                # Initialize with default source/target, can be overridden in translate()
                self.client = GoogleTranslator(source='auto', target='en')
                self.service = 'deep-translator'
                self.logger("Using deep-translator (free Google Translate API) for titles")
                return
            except Exception as e:
                self.logger(f"ERROR: Could not initialize deep-translator: {type(e).__name__}: {e}")
                import traceback
                self.logger(f"Traceback: {traceback.format_exc()}")
        
        # No translator available
        self.client = None
    
    def translate(self, text, source_lang='zh-CN', target_lang='en'):
        """Translate text using deep-translator (for titles and fallback)"""
        if not self.client:
            self.logger("Warning: No translator available")
            return text
        
        try:
            return self._translate_deep(text, source_lang, target_lang)
        except Exception as e:
            self.logger(f"Translation error: {e}")
            return text
    
    def _translate_deep(self, text, source, target):
        """Translate using deep-translator with chunking for long texts"""
        max_length = 4500  # Under 5000 limit
        
        # Map language codes if necessary (deep-translator handles standard codes well)
        # zh-CN is standard, deep-translator uses 'zh-CN' or 'zh-TW' usually.
        # Let's ensure source is correct. deep-translator supports 'zh-CN'.
        
        # Update client source/target for this request
        self.client.source = source
        self.client.target = target
        
        if len(text) <= max_length:
            return self.client.translate(text)
        else:
            # Split by paragraphs and group into chunks
            paragraphs = text.split('\n\n')
            translated_paragraphs = []
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                if current_length + len(para) > max_length and current_chunk:
                    # Translate current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    result = self.client.translate(chunk_text)
                    translated_paragraphs.append(result)
                    
                    # Start new chunk
                    current_chunk = [para]
                    current_length = len(para)
                else:
                    current_chunk.append(para)
                    current_length += len(para)
            
            # Translate remaining chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                result = self.client.translate(chunk_text)
                translated_paragraphs.append(result)
            
            return '\n\n'.join(translated_paragraphs)
