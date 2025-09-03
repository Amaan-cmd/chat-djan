#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add Missing Fields - Batch add common missing fields
"""
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from chunk_manager import ChunkManager

def add_common_fields():
    """Add commonly requested fields for document 7908419"""
    
    manager = ChunkManager()
    
    # Fields that users commonly ask about but might be missing
    missing_fields = [
        {"pdf_id": "7908419", "field": "EMD Required", "value": "No", "page": 2},
        {"pdf_id": "7908419", "field": "ePBG Required", "value": "No", "page": 2},
        {"pdf_id": "7908419", "field": "Evaluation Method", "value": "Total value wise evaluation", "page": 2},
        {"pdf_id": "7908419", "field": "Arbitration Clause", "value": "No", "page": 2},
        {"pdf_id": "7908419", "field": "Mediation Clause", "value": "No", "page": 2},
        {"pdf_id": "7908419", "field": "Payment Timeline", "value": "90 days of issue of consignee receipt-cum-acceptance certificate", "page": 2},
        {"pdf_id": "7908419", "field": "Technical Clarification Time", "value": "2 Days", "page": 2},
        {"pdf_id": "7908419", "field": "Bid to RA enabled", "value": "Yes", "page": 2},
        {"pdf_id": "7908419", "field": "RA Qualification Rule", "value": "H1-Highest Priced Bid Elimination", "page": 2},
        {"pdf_id": "7908419", "field": "MSE Purchase Preference", "value": "Yes", "page": 3},
        {"pdf_id": "7908419", "field": "MII Purchase Preference", "value": "Yes", "page": 3},
    ]
    
    print("=== ADDING MISSING FIELDS FOR DOCUMENT 7908419 ===\n")
    
    if manager.add_multiple_chunks(missing_fields):
        manager.save_index()
        
        # Test some fields
        print("\n=== TESTING ADDED FIELDS ===")
        manager.test_chunk("EMD required 7908419", "7908419")
        manager.test_chunk("evaluation method 7908419", "7908419")
        manager.test_chunk("payment timeline 7908419", "7908419")
        
        print(f"\n✅ Successfully added {len(missing_fields)} missing fields!")
        print("Your chatbot can now answer these questions accurately.")

if __name__ == '__main__':
    add_common_fields()