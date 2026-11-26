import numpy as np
import scipy.io
import heapq
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
    
    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None
        
        def __lt__(self, other):
            return self.freq < other.freq
        
        def __eq__(self, other):
            if not isinstance(other, HeapNode):
                return False
            return self.freq == other.freq
    
    def analyze_mat_file(self, filepath):
        """Analyze the contents of a .mat file"""
        try:
            data = scipy.io.loadmat(filepath)
            print("\n" + "="*50)
            print(f"Analysis of {os.path.basename(filepath)}")
            print("="*50)
            
            # File size information
            original_size = os.path.getsize(filepath)
            print(f"Original file size: {original_size:,} bytes")
            
            # Show variables in the file
            print("\nVariables in the file:")
            for key in data.keys():
                if not key.startswith('__'):
                    value = data[key]
                    print(f"- {key}: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"  Shape: {value.shape}, Data type: {value.dtype}")
                        print(f"  Memory usage: {value.nbytes:,} bytes")
                        if value.size > 0:
                            print(f"  Value range: {np.min(value)} to {np.max(value)}")
            
            # Check if file is suitable for compression
            if original_size < 11000:
                print("\nWarning: File size is less than 11,000 bytes. Compression may not be effective.")
            else:
                print("\nFile meets the size requirement (>11,000 bytes) for effective compression.")
            
            return data
        
        except Exception as e:
            print(f"\nError loading .mat file: {str(e)}")
            return None
    
    def frequency_dict(self, data_bytes):
        """Create frequency dictionary from bytes"""
        frequency = defaultdict(int)
        for byte in data_bytes:
            frequency[byte] += 1
        return frequency
    
    def build_heap(self, frequency):
        """Build priority queue"""
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)
    
    def merge_nodes(self):
        """Build Huffman tree by merging nodes"""
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            
            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            
            heapq.heappush(self.heap, merged)
    
    def make_codes_helper(self, root, current_code):
        """Helper function to build Huffman codes"""
        if root is None:
            return
        
        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return
        
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")
    
    def make_codes(self):
        """Build Huffman codes"""
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)
    
    def get_encoded_data(self, data_bytes):
        """Encode the data using Huffman codes"""
        encoded_data = ""
        for byte in data_bytes:
            encoded_data += self.codes[byte]
        return encoded_data
    
    def pad_encoded_data(self, encoded_data):
        """Pad encoded data to make it multiple of 8"""
        extra_padding = 8 - len(encoded_data) % 8
        for _ in range(extra_padding):
            encoded_data += "0"
        
        padded_info = "{0:08b}".format(extra_padding)
        encoded_data = padded_info + encoded_data
        return encoded_data
    
    def get_byte_array(self, padded_encoded_data):
        """Convert bits to bytes"""
        if len(padded_encoded_data) % 8 != 0:
            print("Encoded data not padded properly")
            exit(0)
        
        b = bytearray()
        for i in range(0, len(padded_encoded_data), 8):
            byte = padded_encoded_data[i:i+8]
            b.append(int(byte, 2))
        return b
    
    def compress(self, input_path, output_path):
        """Main compression function"""
        filename, file_extension = os.path.splitext(input_path)
        
        try:
            # Load .mat file
            mat_data = scipy.io.loadmat(input_path)
            
            # Convert data to bytes (serialize)
            # Note: This simple approach works for numeric arrays
            # For complex .mat files, you might need a more sophisticated serialization
            if len(mat_data) == 0:
                raise ValueError("MAT file contains no data")
            
            # Find the first non-meta variable
            data_key = None
            for key in mat_data.keys():
                if not key.startswith('__'):
                    data_key = key
                    break
            
            if data_key is None:
                raise ValueError("No valid data found in MAT file")
            
            data_bytes = mat_data[data_key].tobytes()
            
            # Check if data is large enough
            if len(data_bytes) < 11000:
                print("Warning: Data size is less than 11,000 bytes. Compression may not be optimal.")
            
            # Huffman coding steps
            frequency = self.frequency_dict(data_bytes)
            self.build_heap(frequency)
            self.merge_nodes()
            self.make_codes()
            encoded_data = self.get_encoded_data(data_bytes)
            padded_encoded_data = self.pad_encoded_data(encoded_data)
            byte_array = self.get_byte_array(padded_encoded_data)
            
            # Prepare output (include both compressed data and Huffman codes)
            output = {
                'compressed_data': byte_array,
                'huffman_codes': self.codes,
                'original_shape': mat_data[data_key].shape,
                'original_dtype': str(mat_data[data_key].dtype),
                'original_key': data_key
            }
            
            # Save compressed file
            with open(output_path, 'wb') as output_file:
                np.savez_compressed(output_file, **output)
            
            # Calculate compression ratio
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            ratio = original_size / compressed_size
            
            print(f"\nCompression complete!")
            print(f"Original size: {original_size:,} bytes")
            print(f"Compressed size: {compressed_size:,} bytes")
            print(f"Compression ratio: {ratio:.2f}:1")
            
            return output_path
        
        except Exception as e:
            print(f"Compression failed: {str(e)}")
            return None
    
    def decompress(self, input_path, output_path):
        """Decompress Huffman encoded file back to .mat"""
        try:
            # Load compressed data
            with np.load(input_path, allow_pickle=True) as data:
                compressed_data = data['compressed_data']
                huffman_codes = data['huffman_codes'].item()
                original_shape = data['original_shape']
                original_dtype = str(data['original_dtype'])
                original_key = str(data['original_key'])
            
            # Convert byte array back to bit string
            bit_string = ""
            for byte in compressed_data:
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
            
            # Remove padding
            padded_info = bit_string[:8]
            extra_padding = int(padded_info, 2)
            bit_string = bit_string[8:-extra_padding] if extra_padding > 0 else bit_string[8:]
            
            # Decode the bits
            reverse_mapping = {v: k for k, v in huffman_codes.items()}
            current_code = ""
            decoded_bytes = bytearray()
            
            for bit in bit_string:
                current_code += bit
                if current_code in reverse_mapping:
                    byte = reverse_mapping[current_code]
                    decoded_bytes.append(byte)
                    current_code = ""
            
            # Convert back to original format
            decoded_array = np.frombuffer(decoded_bytes, dtype=original_dtype).reshape(original_shape)
            
            # Reconstruct .mat file
            mat_data = {original_key: decoded_array}
            scipy.io.savemat(output_path, mat_data)
            
            print(f"\nDecompression complete! File saved to {output_path}")
            return output_path
        
        except Exception as e:
            print(f"Decompression failed: {str(e)}")
            return None

    def get_encoded_bits(self, compressed_data):
        """Convert compressed bytearray back to the original bitstring (without padding)"""
        bitstring = ""
        for byte in compressed_data:
            bitstring += bin(byte)[2:].rjust(8, '0')
        
        # Remove padding info (first 8 bits) and extra padding
        padded_info = bitstring[:8]
        extra_padding = int(padded_info, 2)
        encoded_bits = bitstring[8:-extra_padding] if extra_padding > 0 else bitstring[8:]
        return encoded_bits

    def compare_compressed_files(self, file1_path, file2_path):
        """Compare two compressed .npz files and return similarity percentage"""
        try:
            # Load both compressed files
            with np.load(file1_path, allow_pickle=True) as file1, \
                 np.load(file2_path, allow_pickle=True) as file2:
                
                # Get compressed data (bytearrays)
                compressed_data1 = file1['compressed_data']
                compressed_data2 = file2['compressed_data']
                
                # Extract original bitstrings (without padding)
                bits1 = self.get_encoded_bits(compressed_data1)
                bits2 = self.get_encoded_bits(compressed_data2)
                
                # Compute similarity (using Hamming distance)
                max_len = max(len(bits1), len(bits2))
                min_len = min(len(bits1), len(bits2))
                
                # Pad shorter bitstring with zeros (for alignment)
                bits1 = bits1.ljust(max_len, '0')
                bits2 = bits2.ljust(max_len, '0')
                
                # Calculate matching bits
                matches = sum(b1 == b2 for b1, b2 in zip(bits1, bits2))
                similarity = (matches / max_len) * 100
                
                print(f"\nComparison Results:")
                print(f"- File 1: {os.path.basename(file1_path)} ({len(bits1)} bits)")
                print(f"- File 2: {os.path.basename(file2_path)} ({len(bits2)} bits)")
                print(f"- Similarity: {similarity:.2f}%")
                
                return similarity
        
        except Exception as e:
            print(f"Comparison failed: {str(e)}")
            return None

def plot_comparison(original_size, compressed_size, filename):
    """Plot size comparison before and after compression"""
    plt.figure(figsize=(8, 5))
    labels = ['Original', 'Compressed']
    sizes = [original_size, compressed_size]
    colors = ['#1f77b4', '#ff7f0e']
    
    plt.bar(labels, sizes, color=colors)
    plt.ylabel('Size (bytes)')
    plt.title(f'File Size Comparison\n{filename}')
    
    # Add value labels
    for i, v in enumerate(sizes):
        plt.text(i, v + max(sizes)*0.05, f"{v:,} bytes", ha='center')
    
    plt.tight_layout()
    plt.show()

def main():
    huffman = HuffmanCoding()
    
    print("MAT File Compression using Huffman Coding")
    print("="*50)
    
    while True:
        print("\nMenu:")
        print("1. Analyze .mat file")
        print("2. Compress .mat file")
        print("3. Decompress file")
        print("4. Compare two compressed files")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            # Analyze file
            filepath = input("Enter path to .mat file to analyze: ").strip('"')
            if not os.path.exists(filepath):
                print("File not found!")
                continue
            
            huffman.analyze_mat_file(filepath)
        
        elif choice == '2':
            # Compress file
            input_path = input("Enter path to .mat file to compress: ").strip('"')
            if not os.path.exists(input_path):
                print("File not found!")
                continue
            
            output_path = os.path.splitext(input_path)[0] + "_compressed.npz"
            compressed_path = huffman.compress(input_path, output_path)
            
            if compressed_path:
                original_size = os.path.getsize(input_path)
                compressed_size = os.path.getsize(compressed_path)
                plot_comparison(original_size, compressed_size, os.path.basename(input_path))
        
        elif choice == '3':
            # Decompress file
            input_path = input("Enter path to compressed file (.npz): ").strip('"')
            if not os.path.exists(input_path):
                print("File not found!")
                continue
            
            output_path = os.path.splitext(input_path)[0] + "_decompressed.mat"
            huffman.decompress(input_path, output_path)
        
        elif choice == '4':
            # Compare two compressed files
            file1 = input("Enter first compressed file (.npz): ").strip('"')
            file2 = input("Enter second compressed file (.npz): ").strip('"')
            
            if not os.path.exists(file1) or not os.path.exists(file2):
                print("One or both files not found!")
                continue
            
            huffman.compare_compressed_files(file1, file2)
        
        elif choice == '5':
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()