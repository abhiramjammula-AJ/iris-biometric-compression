import numpy as np
import scipy.io
import heapq
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cdist
from scipy.stats import norm
import random

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

    def enhanced_lrp(self, frequency, window_size=5, alpha=1e-4, beta=0.75, k=1, random_projection_dim=32):
        """Enhanced Local Response Pooling with Random Projection"""
        sorted_freq = sorted(frequency.items(), key=lambda x: x[1])
        chars = [x[0] for x in sorted_freq]
        freqs = np.array([x[1] for x in sorted_freq], dtype=np.float32)
        
        original_total = np.sum(freqs)
        
        if random_projection_dim < len(freqs):
            projection_matrix = np.random.randn(len(freqs), random_projection_dim)
            projected_freqs = np.dot(freqs.T, projection_matrix)
            reconstructed_freqs = np.dot(projected_freqs, projection_matrix.T)
            freqs = 0.7 * freqs + 0.3 * reconstructed_freqs
        
        n = len(freqs)
        new_freqs = np.zeros_like(freqs)
        
        for i in range(n):
            adaptive_window = max(3, window_size - int(i / (n / (window_size // 2))))
            j = max(0, i - adaptive_window // 2)
            upper = min(n, i + adaptive_window // 2 + 1)
            
            local_sum = np.sum(freqs[j:upper]**2)
            new_freqs[i] = freqs[i] / (k + alpha * local_sum)**beta
        
        new_total = np.sum(new_freqs)
        if new_total > 0:
            new_freqs = new_freqs * (original_total / new_total)
        
        lrp_frequency = {chars[i]: int(round(new_freqs[i])) for i in range(n)}
        
        for char in lrp_frequency:
            if lrp_frequency[char] < 1:
                lrp_frequency[char] = 1
                
        return lrp_frequency

    def calculate_eer(self, genuine_scores, impostor_scores):
        """Calculate Equal Error Rate"""
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        threshold = thresholds[eer_idx]
        
        return eer, threshold, fpr, tpr, roc_auc

    def plot_eer_curve(self, genuine_scores, impostor_scores):
        """Plot ROC curve with EER point and proper styling"""
        eer, threshold, fpr, tpr, roc_auc = self.calculate_eer(genuine_scores, impostor_scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.scatter(eer, 1-eer, color='red', s=100, 
                   label=f'EER = {eer:.3f}\nThreshold = {threshold:.3f}')
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        # Annotate EER point
        plt.annotate(f'FPR={eer:.3f}\nTPR={1-eer:.3f}',
                    xy=(eer, 1-eer), xytext=(eer+0.1, 1-eer-0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return eer, threshold

    def analyze_compression_performance(self, original_files, compressed_files):
        """Analyze performance with clear comparison labeling"""
        genuine_scores = []
        impostor_scores = []
        genuine_comparisons = []
        impostor_comparisons = []
        
        print("\n\033[1mGenuine Comparisons (Original vs Compressed):\033[0m")
        print("="*60)
        for i, (orig, comp) in enumerate(zip(original_files, compressed_files)):
            with np.load(comp, allow_pickle=True) as data:
                compressed_data = data['compressed_data']
                bits = self.get_encoded_bits(compressed_data)
                score = len(bits)
                genuine_scores.append(score)
                genuine_comparisons.append((os.path.basename(orig), os.path.basename(comp), score))
                print(f"{i+1:>2}. GENUINE: {os.path.basename(orig):<30} -> {os.path.basename(comp):<30} Score: {score:>8}")

        print("\n\033[1mImpostor Comparisons (Different Files):\033[0m")
        print("="*60)
        comparison_num = 1
        for i in range(len(compressed_files)):
            for j in range(i+1, len(compressed_files)):
                with np.load(compressed_files[i], allow_pickle=True) as data1, \
                     np.load(compressed_files[j], allow_pickle=True) as data2:
                    bits1 = self.get_encoded_bits(data1['compressed_data'])
                    bits2 = self.get_encoded_bits(data2['compressed_data'])
                    score = abs(len(bits1) - len(bits2))
                    impostor_scores.append(score)
                    impostor_comparisons.append((
                        os.path.basename(compressed_files[i]), 
                        os.path.basename(compressed_files[j]), 
                        score))
                    print(f"{comparison_num:>2}. IMPOSTOR: {os.path.basename(compressed_files[i]):<30} vs {os.path.basename(compressed_files[j]):<30} Score: {score:>8}")
                    comparison_num += 1
        
        max_score = max(genuine_scores + impostor_scores)
        genuine_scores_norm = np.array(genuine_scores) / max_score
        impostor_scores_norm = 1 - (np.array(impostor_scores) / max_score)
        
        eer, threshold = self.plot_eer_curve(genuine_scores_norm, impostor_scores_norm)
        
        print("\n\033[1mPerformance Summary:\033[0m")
        print("="*60)
        print(f"{'Total Genuine Comparisons:':<30} {len(genuine_comparisons):>10}")
        print(f"{'Total Impostor Comparisons:':<30} {len(impostor_comparisons):>10}")
        print(f"{'Equal Error Rate (EER):':<30} {eer:>10.3%}")
        print(f"{'Optimal Threshold:':<30} {threshold:>10.4f}")
        
        return eer, threshold, genuine_comparisons, impostor_comparisons

    def analyze_mat_file(self, filepath):
        """Analyze MAT file contents"""
        try:
            data = scipy.io.loadmat(filepath)
            print("\n" + "="*50)
            print(f"Analysis of {os.path.basename(filepath)}")
            print("="*50)

            original_size = os.path.getsize(filepath)
            print(f"Original file size: {original_size:,} bytes")

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

    def build_heap(self, frequency, use_lrp=False):
        """Build priority queue"""
        if use_lrp:
            frequency = self.enhanced_lrp(frequency)

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

    def compress(self, input_path, output_path, use_lrp=False):
        """Main compression function"""
        filename, file_extension = os.path.splitext(input_path)

        try:
            mat_data = scipy.io.loadmat(input_path)

            if len(mat_data) == 0:
                raise ValueError("MAT file contains no data")

            data_key = None
            for key in mat_data.keys():
                if not key.startswith('__'):
                    data_key = key
                    break

            if data_key is None:
                raise ValueError("No valid data found in MAT file")

            data_bytes = mat_data[data_key].tobytes()

            if len(data_bytes) < 11000:
                print("Warning: Data size is less than 11,000 bytes. Compression may not be optimal.")

            frequency = self.frequency_dict(data_bytes)
            self.build_heap(frequency, use_lrp=use_lrp)
            self.merge_nodes()
            self.make_codes()
            encoded_data = self.get_encoded_data(data_bytes)
            padded_encoded_data = self.pad_encoded_data(encoded_data)
            byte_array = self.get_byte_array(padded_encoded_data)

            output = {
                'compressed_data': byte_array,
                'huffman_codes': self.codes,
                'original_shape': mat_data[data_key].shape,
                'original_dtype': str(mat_data[data_key].dtype),
                'original_key': data_key,
                'used_lrp': use_lrp
            }

            with open(output_path, 'wb') as output_file:
                np.savez_compressed(output_file, **output)

            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            ratio = original_size / compressed_size

            print(f"\nCompression complete!")
            print(f"Original size: {original_size:,} bytes")
            print(f"Compressed size: {compressed_size:,} bytes")
            print(f"Compression ratio: {ratio:.2f}:1")
            if use_lrp:
                print("Note: Used enhanced LRP for frequency normalization")

            return output_path

        except Exception as e:
            print(f"Compression failed: {str(e)}")
            return None

    def decompress(self, input_path, output_path):
        """Decompress Huffman encoded file"""
        try:
            with np.load(input_path, allow_pickle=True) as data:
                compressed_data = data['compressed_data']
                huffman_codes = data['huffman_codes'].item()
                original_shape = data['original_shape']
                original_dtype = str(data['original_dtype'])
                original_key = str(data['original_key'])
                used_lrp = data.get('used_lrp', False)

            bit_string = ""
            for byte in compressed_data:
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits

            padded_info = bit_string[:8]
            extra_padding = int(padded_info, 2)
            bit_string = bit_string[8:-extra_padding] if extra_padding > 0 else bit_string[8:]

            reverse_mapping = {v: k for k, v in huffman_codes.items()}
            current_code = ""
            decoded_bytes = bytearray()

            for bit in bit_string:
                current_code += bit
                if current_code in reverse_mapping:
                    byte = reverse_mapping[current_code]
                    decoded_bytes.append(byte)
                    current_code = ""

            decoded_array = np.frombuffer(decoded_bytes, dtype=original_dtype).reshape(original_shape)

            mat_data = {original_key: decoded_array}
            scipy.io.savemat(output_path, mat_data)

            print(f"\nDecompression complete! File saved to {output_path}")
            if used_lrp:
                print("Note: Original file was compressed using enhanced LRP normalization")
            return output_path

        except Exception as e:
            print(f"Decompression failed: {str(e)}")
            return None

    def get_encoded_bits(self, compressed_data):
        """Convert compressed data to bitstring"""
        bitstring = ""
        for byte in compressed_data:
            bitstring += bin(byte)[2:].rjust(8, '0')

        padded_info = bitstring[:8]
        extra_padding = int(padded_info, 2)
        encoded_bits = bitstring[8:-extra_padding] if extra_padding > 0 else bitstring[8:]
        return encoded_bits

    def compare_compressed_files(self, file1_path, file2_path):
        """Compare two compressed files"""
        try:
            with np.load(file1_path, allow_pickle=True) as file1, \
                 np.load(file2_path, allow_pickle=True) as file2:

                compressed_data1 = file1['compressed_data']
                compressed_data2 = file2['compressed_data']

                bits1 = self.get_encoded_bits(compressed_data1)
                bits2 = self.get_encoded_bits(compressed_data2)

                max_len = max(len(bits1), len(bits2))
                min_len = min(len(bits1), len(bits2))

                bits1 = bits1.ljust(max_len, '0')
                bits2 = bits2.ljust(max_len, '0')

                matches = sum(b1 == b2 for b1, b2 in zip(bits1, bits2))
                similarity = (matches / max_len) * 100

                print(f"\nComparison Results:")
                print(f"- File 1: {os.path.basename(file1_path)} ({len(bits1)} bits)")
                print(f"- File 2: {os.path.basename(file2_path)} ({len(bits2)} bits)")
                print(f"- Similarity: {similarity:.2f}%")

                lrp1 = file1.get('used_lrp', False)
                lrp2 = file2.get('used_lrp', False)
                print(f"- File 1 used LRP: {'Yes' if lrp1 else 'No'}")
                print(f"- File 2 used LRP: {'Yes' if lrp2 else 'No'}")

                return similarity

        except Exception as e:
            print(f"Comparison failed: {str(e)}")
            return None

def plot_comparison(original_size, compressed_size, filename):
    """Plot file size comparison with improved styling"""
    plt.figure(figsize=(8, 6))
    labels = ['Original', 'Compressed']
    sizes = [original_size, compressed_size]
    colors = ['#2c7bb6', '#d7191c']
    
    bars = plt.bar(labels, sizes, color=colors, width=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.02,
                f'{height:,} bytes',
                ha='center', va='bottom', fontsize=10)
    
    plt.ylabel('File Size (bytes)', fontsize=12)
    plt.title(f'Compression Ratio: {original_size/compressed_size:.2f}x\n{filename}', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    huffman = HuffmanCoding()

    print("\n\033[1mEnhanced MAT File Compression with EER Evaluation\033[0m")
    print("="*60)

    while True:
        print("\n\033[1mMenu:\033[0m")
        print("1. Analyze .mat file")
        print("2. Compress .mat file")
        print("3. Decompress file")
        print("4. Compare two compressed files")
        print("5. Evaluate compression performance (EER)")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            filepath = input("Enter path to .mat file to analyze: ").strip('"')
            if not os.path.exists(filepath):
                print("File not found!")
                continue
            huffman.analyze_mat_file(filepath)

        elif choice == '2':
            input_path = input("Enter path to .mat file to compress: ").strip('"')
            if not os.path.exists(input_path):
                print("File not found!")
                continue

            use_lrp = input("Use enhanced LRP for frequency normalization? (y/n): ").strip().lower() == 'y'
            output_path = os.path.splitext(input_path)[0] + "_compressed.npz"
            compressed_path = huffman.compress(input_path, output_path, use_lrp=use_lrp)

            if compressed_path:
                original_size = os.path.getsize(input_path)
                compressed_size = os.path.getsize(compressed_path)
                plot_comparison(original_size, compressed_size, os.path.basename(input_path))

        elif choice == '3':
            input_path = input("Enter path to compressed file (.npz): ").strip('"')
            if not os.path.exists(input_path):
                print("File not found!")
                continue

            output_path = os.path.splitext(input_path)[0] + "_decompressed.mat"
            huffman.decompress(input_path, output_path)

        elif choice == '4':
            file1 = input("Enter first compressed file (.npz): ").strip('"')
            file2 = input("Enter second compressed file (.npz): ").strip('"')

            if not os.path.exists(file1) or not os.path.exists(file2):
                print("One or both files not found!")
                continue

            huffman.compare_compressed_files(file1, file2)

        elif choice == '5':
            num_files = int(input("How many files to evaluate? (2-10): "))
            if num_files < 2 or num_files > 10:
                print("Please choose between 2 and 10 files.")
                continue
                
            original_files = []
            compressed_files = []
            
            for i in range(num_files):
                orig = input(f"Enter path to original file {i+1}: ").strip('"')
                comp = os.path.splitext(orig)[0] + "_compressed.npz"
                
                if not os.path.exists(orig):
                    print(f"File not found: {orig}")
                    break
                    
                if not os.path.exists(comp):
                    print(f"Compressed file not found for {orig}")
                    print("Would you like to compress it now? (y/n)")
                    if input().strip().lower() == 'y':
                        use_lrp = input("Use enhanced LRP? (y/n): ").strip().lower() == 'y'
                        huffman.compress(orig, comp, use_lrp=use_lrp)
                    else:
                        break
                
                original_files.append(orig)
                compressed_files.append(comp)
            
            if len(original_files) == num_files:
                eer, threshold, genuine, impostor = huffman.analyze_compression_performance(original_files, compressed_files)
                
                print("\n\033[1mDetailed Results:\033[0m")
                print("\nGenuine Comparisons:")
                for comp in genuine:
                    print(f"- {comp[0]} -> {comp[1]}: Score = {comp[2]}")
                
                print("\nImpostor Comparisons:")
                for comp in impostor:
                    print(f"- {comp[0]} vs {comp[1]}: Score = {comp[2]}")

        elif choice == '6':
            print("Exiting program...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()