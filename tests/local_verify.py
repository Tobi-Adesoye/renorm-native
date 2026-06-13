import sys
import torch

print("--- Starting Pure Verification Test ---")

try:
    from renorm.layers import RenormLinear
    print("\n=========================================================")
    print("SUCCESS: RenormLinear was imported perfectly!")
    print("=========================================================")
except ModuleNotFoundError as e:
    print("\n=========================================================")
    print(f"FAILED: The package could not be found. Error: {e}")
    print("=========================================================")