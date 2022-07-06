import re

class TextCleaner:
    def __init__(self, filters=None):
        if filters:
            self.punct = 'punctuation' in filters
            self.action = 'action' in filters
            self.hes = 'hesitation' in filters  
        else:
            self.punct = False
            self.action = False
            self.hes = False
            
    def clean_text(self, text:str)->str:
        """method which cleans text with chosen convention"""
        if self.action:
            text = re.sub("[\[\(\<\%].*?[\]\)\>\%]", "", text)    
        if self.punct: 
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
        if self.hes:
            text = self.hesitation(text)
        text = ' '.join(text.split())
        return text

    @staticmethod
    def hesitation(text:str)->str:
        """internal function to converts hesitation"""
        hes_maps = {"umhum":"um", "uh-huh":"um", 
                    "uhhuh":"um", "hum":"um", "uh":'um'}

        for h1, h2 in hes_maps.items():
            if h1 in text:
                pattern = r'(^|[^a-zA-z])'+h1+r'($|[^a-zA-Z])'
                text = re.sub(pattern, r'\1'+h2+r'\2', text)
                #run line twice as uh uh share middle character
                text = re.sub(pattern, r'\1'+h2+r'\2', text)
        return text
