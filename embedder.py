"""
CTCAE multi-toxicity extractor and matcher.
This script loads CTCAE terms from CSV, extracts multiple potential toxicities from a clinical note,
and matches each toxicity to the most appropriate CTCAE term.

Example usage:
    # Setup the database and run with example
    python organized_multi_toxicity_embedder.py --setup --example
    
    # Analyze a specific clinical note
    python organized_multi_toxicity_embedder.py --note "Patient has hemoglobin of 9.0 g/dL with fatigue. Also experiencing grade 2 nausea after chemotherapy."
    
    # Interactive mode
    python organized_multi_toxicity_embedder.py
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
import sqlite3
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CTCAEMultiToxicityExtractor:
    def __init__(self, csv_path: str, db_path: str = "ctcae.sqlite", setup: bool = False):
        """
        Initialize the extractor with a CSV file containing CTCAE terms
        
        Args:
            csv_path: Path to the CTCAE CSV file
            db_path: Path to SQLite database
            setup: Whether to set up the database from scratch
        """
        self.csv_path = csv_path
        self.db_path = db_path
        
        # Load model and tokenizer
        print(f"Loading model on {DEVICE}...")
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        
        # Load OpenAI client for GPT-4o-mini
        self.client = OpenAI()
        
        # Set up SQLite database
        if setup or not os.path.exists(db_path):
            self._setup_database()
        else:
            # Connect to existing database
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
    
    def _setup_database(self):
        """Set up SQLite database with vector extension"""
        print(f"Setting up SQLite database at {self.db_path}...")
        
        # Load CTCAE data
        print(f"Loading CTCAE data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        # Connect to database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self._create_database_tables()
        
        # Process CTCAE terms and add to database
        self._populate_database()
                
        self.conn.commit()
        print("Database setup complete!")
    
    def _create_database_tables(self):
        """Create necessary database tables"""
        # Main table for CTCAE terms
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS ctcae_terms (
            id INTEGER PRIMARY KEY,
            meddra_code TEXT,
            meddra_soc TEXT,
            ctcae_term TEXT,
            grade1 TEXT,
            grade2 TEXT,
            grade3 TEXT,
            grade4 TEXT,
            grade5 TEXT,
            definition TEXT,
            embedding TEXT
        )
        ''')
        
        # Table for system organ classes
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_organ_classes (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE
        )
        ''')
        
        self.conn.commit()
    
    def _populate_database(self):
        """Populate database with CTCAE terms and their embeddings"""
        print(f"Processing {len(self.df)} CTCAE terms...")
        
        # Print the column names for debugging
        print(f"CSV column names: {self.df.columns.tolist()}")
        
        # Fix: Handle possible whitespace in column names
        # Some CSV files have column names with extra spaces like 'Grade 1   '
        clean_columns = {}
        for col in self.df.columns:
            clean_col = col.strip()
            if clean_col != col:
                clean_columns[col] = clean_col
        
        if clean_columns:
            print(f"Cleaning whitespace from column names: {clean_columns}")
            self.df = self.df.rename(columns=clean_columns)
        
        for idx, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
            self.idx = idx  # Store idx for debugging in _insert_term_into_database
            
            # Create text representation for embedding
            text = self._create_term_text_for_embedding(row)
            
            # Create embedding
            embedding = self._embed_text(text)
            
            # Insert into database
            self._insert_term_into_database(row, embedding)
            
            # Add system organ class if not already in the table
            soc = row.get('MedDRA SOC', '')
            if soc:
                self.cursor.execute(
                    'INSERT OR IGNORE INTO system_organ_classes (name) VALUES (?)',
                    (soc,)
                )
        
        # Debug: Print sample data from the database
        self.cursor.execute('SELECT ctcae_term, grade1, grade2, grade3, grade4, grade5 FROM ctcae_terms LIMIT 5')
        sample_data = self.cursor.fetchall()
        print("Sample data from database:")
        for row in sample_data:
            print(f"CTCAE Term: {row[0]}")
            print(f"  Grade 1: '{row[1]}'")
            print(f"  Grade 2: '{row[2]}'")
            print(f"  Grade 3: '{row[3]}'")
            print(f"  Grade 4: '{row[4]}'")
            print(f"  Grade 5: '{row[5]}'")
            print()
    
    def _create_term_text_for_embedding(self, row: pd.Series) -> str:
        """Create text representation of a CTCAE term for embedding"""
        text = f"{row['CTCAE Term']}. {row['Definition']}"
        
        # Include grades information if available
        # Fix: Handle grade columns with possible whitespace or different formats
        grades = []
        for i in range(1, 6):
            # Look for columns that may contain Grade i information
            grade_cols = [col for col in row.index if f'Grade {i}' in col or f'Grade{i}' in col]
            
            if grade_cols:
                grade_col = grade_cols[0]  # Use the first matching column
                if pd.notna(row[grade_col]) and row[grade_col]:
                    grade_text = row[grade_col].strip()
                    if grade_text:
                        grades.append(f"Grade {i}: {grade_text}")
        
        if grades:
            text += " " + " ".join(grades)
            
        return text
    
    def _insert_term_into_database(self, row: pd.Series, embedding: np.ndarray):
        """Insert a CTCAE term and its embedding into the database"""
        # Fix: Properly extract grade information from CSV
        # The CSV has columns with spaces like 'Grade 1   ' - we need to handle that
        grade_cols = [col for col in row.index if 'Grade' in col]
        
        grade1 = row.get(next((col for col in grade_cols if '1' in col), ''), '').strip()
        grade2 = row.get(next((col for col in grade_cols if '2' in col), ''), '').strip()
        grade3 = row.get(next((col for col in grade_cols if '3' in col), ''), '').strip()
        grade4 = row.get(next((col for col in grade_cols if '4' in col), ''), '').strip()
        grade5 = row.get(next((col for col in grade_cols if '5' in col), ''), '').strip()
        
        if hasattr(self, 'idx') and self.idx <= 3 or (not hasattr(self, 'idx') and row.name <= 3):  # Debug first few entries
            print(f"Inserting {row.get('CTCAE Term', '')}")
            print(f"  Grade 1: '{grade1}'")
            print(f"  Grade 2: '{grade2}'")
            print(f"  Grade 3: '{grade3}'")
            print(f"  Grade 4: '{grade4}'")
            print(f"  Grade 5: '{grade5}'")
        
        self.cursor.execute('''
        INSERT INTO ctcae_terms 
        (meddra_code, meddra_soc, ctcae_term, grade1, grade2, grade3, grade4, grade5, definition, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(row.get('MedDRA Code', '')),
            row.get('MedDRA SOC', ''),
            row.get('CTCAE Term', ''),
            grade1,
            grade2,
            grade3,
            grade4,
            grade5,
            row.get('Definition', ''),
            json.dumps(embedding.tolist())
        ))
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for a text
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        with torch.no_grad():
            # Tokenize and encode
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=512,
            ).to(DEVICE)
            
            # Get embedding from [CLS] token
            embed = self.model(**encoded).last_hidden_state[:, 0, :]
            
            # Convert to numpy and normalize
            embed_np = embed[0].cpu().numpy()
            # Normalize to unit length for cosine similarity
            embed_np = embed_np / np.linalg.norm(embed_np)
            
            return embed_np
    
    def extract_toxicities(self, clinical_note: str) -> List[Dict[str, Any]]:
        """
        Extract potential toxicities from a clinical note using GPT
        
        Args:
            clinical_note: Clinical note to extract toxicities from
            
        Returns:
            List of dictionaries with extracted toxicities
        """
        # Get list of system organ classes from database for reference
        self.cursor.execute('SELECT name FROM system_organ_classes')
        system_organ_classes = [row[0] for row in self.cursor.fetchall()]
        
        # Format prompt for GPT
        prompt = self._create_extraction_prompt(clinical_note, system_organ_classes)
        
        # Call GPT-4o-mini
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extract toxicities from the result
            toxicities = result.get("toxicities", [])
            if not toxicities and isinstance(result, list):
                # Handle case where response is directly an array
                toxicities = result
                
            return toxicities
            
        except Exception as e:
            print(f"Error calling GPT-4o-mini for toxicity extraction: {e}")
            # Return empty list if there's an error
            return []
    
    def _create_extraction_prompt(self, clinical_note: str, system_organ_classes: List[str]) -> str:
        """Create prompt for toxicity extraction"""
        return f"""
I need to extract potential toxicities from a clinical note for oncology patients. 
The toxicities should be mapped to CTCAE (Common Terminology Criteria for Adverse Events) terms.

Clinical note: 
{clinical_note}

Please identify all potential adverse events or toxicities mentioned in this note. 
For each toxicity:
1. Extract the relevant text describing the symptom or condition
2. Estimate the grade if mentioned (1-5, where 1 is mild and 5 is death)
3. Identify which body system it affects

Here are the CTCAE body systems (MedDRA System Organ Classes) for reference:
{', '.join(system_organ_classes)}

Return your response as a JSON array with each toxicity containing these fields:
- "description": The symptom or condition text from the note
- "grade": The estimated grade (1-5) if mentioned, or null
- "body_system": The most likely body system from the list above
- "context": The full sentence or context in which this toxicity was mentioned

Do not include normal findings or information that does not represent a potential adverse event.
"""
    
    def match_toxicity_to_ctcae(self, toxicity: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """
        Match a toxicity to the most similar CTCAE terms
        
        Args:
            toxicity: Dictionary containing toxicity information
            k: Number of similar terms to retrieve
            
        Returns:
            List of dictionaries containing similar CTCAE terms
        """
        # Create a query text from the toxicity information
        query_text = self._create_query_text(toxicity)
        
        # Create embedding for the query
        query_embedding = self._embed_text(query_text)
        
        # Get database records based on body system
        results = self._get_database_records(toxicity)
        
        # Calculate similarity for each term
        similarities = self._calculate_similarities(toxicity, results, query_embedding)
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:k]
    
    def _create_query_text(self, toxicity: Dict[str, Any]) -> str:
        """Create query text from toxicity information"""
        description = toxicity.get('description', '')
        body_system = toxicity.get('body_system', '')
        grade = toxicity.get('grade')
        context = toxicity.get('context', '')
        
        query_text = f"Description: {description}. "
        if body_system:
            query_text += f"Body system: {body_system}. "
        if grade:
            query_text += f"Grade: {grade}. "
        query_text += f"Context: {context}"
        
        return query_text
    
    def _get_database_records(self, toxicity: Dict[str, Any]) -> List:
        """Get database records based on body system"""
        body_system = toxicity.get('body_system', '')
        
        # Debug: Check which body system we're querying
        print(f"Querying database for body system: '{body_system}'")
        
        if body_system and body_system != "Unknown":
            self.cursor.execute(
                'SELECT id, ctcae_term, definition, grade1, grade2, grade3, grade4, grade5, embedding, meddra_soc, meddra_code FROM ctcae_terms WHERE meddra_soc = ?',
                (body_system,)
            )
        else:
            self.cursor.execute(
                'SELECT id, ctcae_term, definition, grade1, grade2, grade3, grade4, grade5, embedding, meddra_soc, meddra_code FROM ctcae_terms'
            )
        
        results = self.cursor.fetchall()
        
        # Debug: Print number of results and sample grade data
        print(f"Found {len(results)} matches in database")
        if results and len(results) > 0:
            print(f"Sample result: {results[0][1]} - Grade 1: '{results[0][3]}', Grade 2: '{results[0][4]}'")
        
        return results
    
    def _calculate_similarities(self, toxicity: Dict[str, Any], results: List, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate similarities between toxicity and CTCAE terms"""
        similarities = []
        description = toxicity.get('description', '').lower()
        
        for row in results:
            term_id, term, definition, grade1, grade2, grade3, grade4, grade5, embedding_json, meddra_soc, meddra_code = row
            term_embedding = np.array(json.loads(embedding_json))
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, term_embedding)
            
            # Apply term-specific boosts
            similarity = self._apply_term_specific_boosts(description, term.lower(), similarity)
            
            # Create result dictionary
            similarities.append({
                'id': term_id,
                'ctcae_term': term,
                'definition': definition,
                'grade1': grade1,
                'grade2': grade2,
                'grade3': grade3,
                'grade4': grade4,
                'grade5': grade5,
                'meddra_soc': meddra_soc,
                'meddra_code': meddra_code,
                'similarity_score': float(similarity)
            })
            
        return similarities
    
    def _apply_term_specific_boosts(self, description: str, term: str, similarity: float) -> float:
        """Apply term-specific similarity boosts"""
        # For anemia, check specifically if description contains hemoglobin values and match with CTCAE anemia criteria
        if "anemia" in term or "hemoglobin" in description or "hgb" in description:
            hgb_matches = re.findall(r'(\d+\.\d+|\d+)\s*g\/dL', description)
            if hgb_matches:
                try:
                    hgb_value = float(hgb_matches[0])
                    # Add anemia-specific matching score boost
                    if hgb_value < 13.5 and hgb_value >= 10.0:  # Grade 1
                        similarity += 0.2
                    elif hgb_value < 10.0 and hgb_value >= 8.0:  # Grade 2
                        similarity += 0.3
                    elif hgb_value < 8.0:  # Grade 3+
                        similarity += 0.4
                except ValueError:
                    pass
        
        # For nausea/vomiting, boost matching for these terms
        if ("nausea" in description and "nausea" in term) or \
           ("vomit" in description and "vomit" in term):
            similarity += 0.3
        
        # Similarly for other common terms
        if "pain" in description and "pain" in term:
            similarity += 0.2
        
        if "dysphag" in description and "dysphag" in term:
            similarity += 0.2
            
        if "numb" in description and "paresthesia" in term:
            similarity += 0.2
            
        if "tingl" in description and "paresthesia" in term:
            similarity += 0.2
            
        return similarity
    
    def select_best_match(self, toxicity: Dict[str, Any], similar_terms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best CTCAE term for a toxicity using GPT
        
        Args:
            toxicity: Dictionary containing toxicity information
            similar_terms: List of similar CTCAE terms
            
        Returns:
            The best matching CTCAE term
        """
        # Format prompt for GPT
        prompt = self._create_selection_prompt(toxicity, similar_terms)
        
        # Call GPT-4o-mini
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            print(f"Raw GPT result: {result}")
            
            best_idx = self._process_gpt_selection_result(result, len(similar_terms))
            print(f"Selected term: {similar_terms[best_idx]['ctcae_term']} (index {best_idx})")
            
            # Add reasoning to the result
            similar_terms[best_idx]['gpt_reasoning'] = result.get("reasoning", "No reasoning provided")
            
            # Validate the selection against the reasoning
            reasoning = result.get("reasoning", "").lower()
            selected_term = similar_terms[best_idx]['ctcae_term'].lower()
            
            # Check if the reasoning mentions a different term than the one selected
            for i, term in enumerate(similar_terms):
                term_name = term['ctcae_term'].lower()
                # If the term is mentioned prominently in the reasoning but wasn't selected
                if term_name != selected_term and term_name in reasoning and len(reasoning.split(term_name)) > 2:
                    print(f"WARNING: Reasoning mentions {term_name} but selected {selected_term}. Checking consistency...")
                    
                    # If reasoning strongly implies a different term should be selected
                    if f"should be {term_name}" in reasoning or f"best match is {term_name}" in reasoning or f"appropriate choice is {term_name}" in reasoning:
                        print(f"CORRECTION: Based on reasoning, changing selection from {selected_term} to {term_name}")
                        best_idx = i
            
            # Add the original toxicity information
            similar_terms[best_idx]['extracted_description'] = toxicity.get('description', '')
            similar_terms[best_idx]['extracted_grade'] = toxicity.get('grade')
            similar_terms[best_idx]['extracted_body_system'] = toxicity.get('body_system', '')
            similar_terms[best_idx]['extracted_context'] = toxicity.get('context', '')
            
            return similar_terms[best_idx]
            
        except Exception as e:
            print(f"Error calling GPT-4o-mini for term selection: {e}")
            # Fallback to highest similarity score
            best_term = similar_terms[0]
            best_term['extracted_description'] = toxicity.get('description', '')
            best_term['extracted_grade'] = toxicity.get('grade')
            best_term['extracted_body_system'] = toxicity.get('body_system', '')
            best_term['extracted_context'] = toxicity.get('context', '')
            return best_term
    
    def _create_selection_prompt(self, toxicity: Dict[str, Any], similar_terms: List[Dict[str, Any]]) -> str:
        """Create prompt for term selection"""
        prompt = f"""
I need to select the most appropriate CTCAE (Common Terminology Criteria for Adverse Events) term for a toxicity.

Extracted toxicity:
- Description: {toxicity.get('description', '')}
- Body system: {toxicity.get('body_system', '')}
- Grade: {toxicity.get('grade', 'Not specified')}
- Context: {toxicity.get('context', '')}

Top {len(similar_terms)} candidate CTCAE terms:
"""
        
        for i, term in enumerate(similar_terms):
            prompt += f"""
{i+1}. CTCAE Term: {term['ctcae_term']}
   Body System: {term['meddra_soc']}
   Definition: {term['definition']}
   Similarity Score: {term['similarity_score']:.4f}
   Grades:
"""
            for j in range(1, 6):
                grade_key = f'grade{j}'
                if grade_key in term and term[grade_key] and term[grade_key].strip():
                    prompt += f"      - Grade {j}: {term[grade_key]}\n"
                else:
                    prompt += f"      - Grade {j}: Not defined\n"
        
        prompt += """
Please analyze the extracted toxicity information and select the most clinically appropriate CTCAE term from the list above.
Your selection should be based on medical relevance, not just similarity scores.

For example, descriptions mentioning hemoglobin levels should be matched with anemia-related terms.

Provide your response in this exact JSON format:
{
  "best_match_index": [integer between 1 and the number of terms],
  "reasoning": [detailed explanation of your choice]
}

The best_match_index MUST be the number (1, 2, 3, etc.) corresponding to the term you selected from the list above.
"""
        return prompt
    
    def _process_gpt_selection_result(self, result: Dict, num_terms: int) -> int:
        """Process GPT selection result to get the best match index"""
        best_idx = result.get("best_match_index", 0)
        
        # Debug: Print the raw response from GPT
        print(f"GPT selection response: {result}")
        
        # Adjust index (GPT returns 1-based, we need 0-based)
        if isinstance(best_idx, str) and best_idx.isdigit():
            best_idx = int(best_idx) - 1
            print(f"Converted string index {best_idx+1} to zero-based index {best_idx}")
        elif isinstance(best_idx, int):
            best_idx = best_idx - 1  # Convert from 1-based to 0-based
            print(f"Converted integer index {best_idx+1} to zero-based index {best_idx}")
        else:
            print(f"WARNING: Unexpected index type: {type(best_idx)}, value: {best_idx}")
            best_idx = 0
        
        # Ensure index is within range
        final_idx = max(0, min(num_terms - 1, best_idx))
        if final_idx != best_idx:
            print(f"WARNING: Index {best_idx} out of range, adjusted to {final_idx}")
            
        return final_idx
    
    def process_clinical_note(self, clinical_note: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Process a clinical note to extract and match toxicities
        
        Args:
            clinical_note: Clinical note to process
            k: Number of similar terms to retrieve for each toxicity
            
        Returns:
            List of dictionaries containing matched toxicities
        """
        # Extract potential toxicities
        extracted_toxicities = self.extract_toxicities(clinical_note)
        
        if not extracted_toxicities:
            print("No toxicities extracted from the clinical note.")
            return []
        
        # Match each toxicity to CTCAE terms
        matched_toxicities = []
        for toxicity in extracted_toxicities:
            similar_terms = self.match_toxicity_to_ctcae(toxicity, k)
            if similar_terms:
                best_match = self.select_best_match(toxicity, similar_terms)
                matched_toxicities.append({
                    'toxicity': toxicity,
                    'ctcae_match': best_match,
                    'all_matches': similar_terms[:k]
                })
            else:
                print(f"No matches found for toxicity: {toxicity.get('description', '')}")
        
        return matched_toxicities

    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


def print_toxicity_results(matched_toxicities: List[Dict[str, Any]]):
    """Print the toxicity extraction and matching results"""
    if matched_toxicities:
        print(f"\n=== Extracted {len(matched_toxicities)} Toxicities ===")
        for i, match in enumerate(matched_toxicities):
            toxicity = match['toxicity']
            ctcae_match = match['ctcae_match']
            all_matches = match['all_matches']
            
            # Print extracted toxicity information
            print(f"\n{i+1}. Extracted Toxicity: {toxicity.get('description', '')}")
            print(f"   Body System: {toxicity.get('body_system', '')}")
            print(f"   Estimated Grade: {toxicity.get('grade', 'Not specified')}")
            print(f"   Context: {toxicity.get('context', '')}")
            
            # Show all potential matches
            print(f"\n   Top {len(all_matches)} Matching CTCAE Terms:")
            for j, term in enumerate(all_matches):
                print(f"   {j+1}. {term['ctcae_term']} (Score: {term['similarity_score']:.4f})")
                print(f"      Body System: {term['meddra_soc']}")
                if 'meddra_code' in term:
                    print(f"      MedDRA Code: {term['meddra_code']}")
            
            # Show the best match with all details
            print(f"\n   Best CTCAE Match: {ctcae_match['ctcae_term']}")
            print(f"   Definition: {ctcae_match['definition']}")
            print(f"   CTCAE Body System: {ctcae_match['meddra_soc']}")
            
            # Print all grades for the best match
            print(f"\n   CTCAE Grades for {ctcae_match['ctcae_term']}:")
            for j in range(1, 6):
                grade_key = f'grade{j}'
                if grade_key in ctcae_match and ctcae_match[grade_key] and ctcae_match[grade_key].strip():
                    print(f"      Grade {j}: {ctcae_match[grade_key]}")
                else:
                    print(f"      Grade {j}: Not defined")
            
            # Highlight the matching grade if estimated grade is available
            grade = toxicity.get('grade')
            if grade and isinstance(grade, (int, float)) and 1 <= grade <= 5:
                grade_key = f'grade{int(grade)}'
                if grade_key in ctcae_match and ctcae_match[grade_key] and ctcae_match[grade_key].strip():
                    print(f"\n   Matching Grade {int(grade)} Definition: {ctcae_match[grade_key]}")
            elif isinstance(grade, str) and grade.isdigit() and 1 <= int(grade) <= 5:
                grade_key = f'grade{grade}'
                if grade_key in ctcae_match and ctcae_match[grade_key] and ctcae_match[grade_key].strip():
                    print(f"\n   Matching Grade {grade} Definition: {ctcae_match[grade_key]}")
            
            print(f"\n   Reasoning: {ctcae_match.get('gpt_reasoning', 'No reasoning provided')}")
    else:
        print("\nNo toxicities were extracted from the note.")


def example_usage():
    """Example usage with a sample clinical note containing multiple toxicities"""
    print("\n=== Example Usage ===")
    
    # Sample clinical note with multiple toxicities
    sample_note = """
    Patient presented for follow-up after cycle 2 of FOLFOX chemotherapy. 
    Patient reports experiencing moderate nausea and vomiting for 2 days after chemotherapy, 
    which was partially controlled with ondansetron. 
    Also notes numbness and tingling in fingertips and toes, which has been persistent since last cycle.
    Labs show hemoglobin of 9.2 g/dL (baseline was 13.5 g/dL). 
    Patient also mentions difficulty swallowing solid foods and pain in the mouth.
    Patient denies fever, chills, or signs of infection.
    """
    
    # Create extractor
    extractor = CTCAEMultiToxicityExtractor('ctcae_v5.csv')
    
    # Process the note
    matched_toxicities = extractor.process_clinical_note(sample_note)
    
    # Print results
    print("\n=== Sample Clinical Note ===")
    print(sample_note)
    
    print_toxicity_results(matched_toxicities)
    
    print("\n=== End of Example ===")
    
    # Close database connection
    extractor.close()


def main():
    """Main function to run the extractor from command line"""
    parser = argparse.ArgumentParser(description='CTCAE Multi-Toxicity Extractor and Matcher')
    parser.add_argument('--csv', type=str, default='ctcae_v5.csv', help='Path to CTCAE CSV file')
    parser.add_argument('--note', type=str, help='Clinical note to analyze')
    parser.add_argument('--db', type=str, default='ctcae.sqlite', help='Path to SQLite database')
    parser.add_argument('--k', type=int, default=5, help='Number of similar terms to retrieve')
    parser.add_argument('--setup', action='store_true', help='Set up database from scratch')
    parser.add_argument('--example', action='store_true', help='Run with an example clinical note')
    
    args = parser.parse_args()
    
    if args.example:
        # Run with example note
        example_usage()
        return
    
    extractor = CTCAEMultiToxicityExtractor(args.csv, args.db, args.setup)
    
    if args.note:
        # Process a single note
        matched_toxicities = extractor.process_clinical_note(args.note, args.k)
        
        print("\n=== Clinical Note ===")
        print(args.note)
        
        print_toxicity_results(matched_toxicities)
    else:
        # Interactive mode
        print("\nEnter a clinical note to analyze (type 'exit' to quit):")
        while True:
            note = input("\nClinical note: ")
            if note.lower() == 'exit':
                break
                
            matched_toxicities = extractor.process_clinical_note(note, args.k)
            print_toxicity_results(matched_toxicities)
    
    # Close database connection
    extractor.close()


if __name__ == "__main__":
    main()