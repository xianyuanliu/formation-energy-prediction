# Unique Space Group Description Generator
# Relies on inherent uniqueness of Hermann-Mauguin notation without artificial templating

import json
from pymatgen.symmetry.groups import SpaceGroup
import re
from collections import defaultdict

class SpaceGroupDescriptor:
    def __init__(self):
        # Comprehensive symmetry element analysis patterns
        self.screw_pattern = r'(\d+)_(\d+)'
        self.glide_chars = set('abcdne')
        
    def get_bravais_lattice_description(self, symbol: str) -> str:
        """Concise Bravais lattice description"""
        first_char = symbol[0]
        desc_map = {
            'P': 'Primitive (P) lattice - corner points only',
            'I': 'Body-centered (I) lattice - corners + center',
            'F': 'Face-centered (F) lattice - corners + face centers',
            'A': 'A-face-centered lattice - corners + A faces',
            'B': 'B-face-centered lattice - corners + B faces', 
            'C': 'C-face-centered lattice - corners + C faces',
            'R': 'Rhombohedral (R) lattice - trigonal setting'
        }
        return desc_map.get(first_char, 'Unknown lattice')

    def parse_hermann_mauguin_completely(self, symbol: str, crystal_system: str) -> dict:
        """Complete parsing of Hermann-Mauguin symbol to extract all unique features"""
        # Remove Bravais lattice letter
        symmetry_notation = symbol[1:].strip()
        
        parsed = {
            'symbol': symbol,
            'symmetry_notation': symmetry_notation,
            'has_slash': '/' in symmetry_notation,
            'primary_elements': [],
            'secondary_elements': [],
            'plane_elements': [],
            'special_features': []
        }
        
        if '/' in symmetry_notation:
            # Split at '/' to separate axes from planes
            parts = symmetry_notation.split('/')
            axis_part = parts[0]
            plane_part = parts[1] if len(parts) > 1 else ''
            
            # Analyze axis part
            parsed['axis_part'] = axis_part
            parsed['plane_part'] = plane_part
            
            # Check for screw axis in axis part
            if '_' in axis_part:
                screw_match = re.search(self.screw_pattern, axis_part)
                if screw_match:
                    fold, translation = screw_match.groups()
                    parsed['primary_elements'].append({
                        'type': 'screw_axis',
                        'fold': fold,
                        'translation': translation,
                        'full_notation': f"{fold}_{translation}"
                    })
            elif axis_part.isdigit():
                parsed['primary_elements'].append({
                    'type': 'rotation_axis',
                    'fold': axis_part
                })
            elif axis_part.startswith('-'):
                # Rotoinversion axes like -4, -6
                parsed['primary_elements'].append({
                    'type': 'rotoinversion_axis',
                    'fold': axis_part[1:],
                    'full_notation': axis_part
                })
            
            # Analyze plane part character by character
            for char in plane_part:
                if char == 'm':
                    parsed['plane_elements'].append({
                        'type': 'mirror',
                        'notation': 'm'
                    })
                elif char in self.glide_chars:
                    parsed['plane_elements'].append({
                        'type': 'glide',
                        'glide_type': char,
                        'notation': char
                    })
        
        else:
            # No slash - parse sequentially
            i = 0
            while i < len(symmetry_notation):
                char = symmetry_notation[i]
                
                # Check for screw axis
                if char.isdigit() and i + 1 < len(symmetry_notation) and symmetry_notation[i + 1] == '_':
                    screw_match = re.search(self.screw_pattern, symmetry_notation[i:])
                    if screw_match:
                        fold, translation = screw_match.groups()
                        element_info = {
                            'type': 'screw_axis',
                            'fold': fold,
                            'translation': translation,
                            'full_notation': f"{fold}_{translation}",
                            'position': i
                        }
                        
                        if i == 0:
                            parsed['primary_elements'].append(element_info)
                        else:
                            parsed['secondary_elements'].append(element_info)
                        
                        i += len(screw_match.group(0))
                        continue
                
                # Regular rotation axis
                elif char.isdigit():
                    element_info = {
                        'type': 'rotation_axis',
                        'fold': char,
                        'position': i
                    }
                    
                    if i == 0:
                        parsed['primary_elements'].append(element_info)
                    else:
                        parsed['secondary_elements'].append(element_info)
                
                # Mirror planes
                elif char == 'm':
                    parsed['plane_elements'].append({
                        'type': 'mirror',
                        'notation': 'm',
                        'position': i
                    })
                
                # Glide planes
                elif char in self.glide_chars:
                    parsed['plane_elements'].append({
                        'type': 'glide',
                        'glide_type': char,
                        'notation': char,
                        'position': i
                    })
                
                # Inversion notation
                elif char == '-':
                    parsed['special_features'].append({
                        'type': 'inversion_notation',
                        'position': i
                    })
                
                i += 1
        
        return parsed

    def describe_primary_symmetry(self, parsed: dict, crystal_system: str) -> str:
        """Describe primary symmetry elements based on their actual characteristics"""
        descriptions = []
        
        for element in parsed['primary_elements']:
            if element['type'] == 'rotation_axis':
                fold = element['fold']
                if crystal_system == 'cubic':
                    if fold == '4':
                        descriptions.append("Contains 4-fold rotation axes along cubic <100> directions")
                    elif fold == '3':
                        descriptions.append("Features 3-fold rotation axes along <111> body diagonals")
                    elif fold == '2':
                        descriptions.append("Exhibits 2-fold rotation axes in cubic symmetry")
                        
                elif crystal_system == 'tetragonal':
                    if fold == '4':
                        descriptions.append("Possesses a principal 4-fold rotation axis along the c-direction")
                    elif fold == '2':
                        descriptions.append("Contains 2-fold rotation axes perpendicular to the principal axis")
                        
                elif crystal_system == 'hexagonal':
                    if fold == '6':
                        descriptions.append("Contains a 6-fold rotation axis defining hexagonal symmetry")
                    elif fold == '3':
                        descriptions.append("Features a 3-fold rotation axis as the primary element")
                    elif fold == '2':
                        descriptions.append("Includes 2-fold rotation axes in the hexagonal framework")
                        
                elif crystal_system == 'trigonal':
                    if fold == '3':
                        if 'R' in parsed['symbol']:
                            descriptions.append("Exhibits 3-fold rotation in rhombohedral setting")
                        else:
                            descriptions.append("Contains 3-fold trigonal rotation as the defining element")
                            
                elif crystal_system == 'orthorhombic':
                    descriptions.append(f"Contains {fold}-fold rotation along a principal crystallographic axis")
                    
                elif crystal_system == 'monoclinic':
                    descriptions.append(f"Features a unique {fold}-fold rotation axis as the only rotational symmetry")
                    
            elif element['type'] == 'screw_axis':
                fold = element['fold']
                translation = element['translation']
                full_notation = element['full_notation']
                
                # Calculate the actual translation fraction
                trans_fraction = f"{translation}/{fold}"
                
                descriptions.append(f"Incorporates a {full_notation} screw axis combining {fold}-fold rotation with {trans_fraction} unit cell translation")
                
            elif element['type'] == 'rotoinversion_axis':
                fold = element['fold']
                full_notation = element['full_notation']
                descriptions.append(f"Features a {full_notation} rotoinversion axis combining {fold}-fold rotation with inversion")
        
        return ". ".join(descriptions) if descriptions else ""

    def describe_secondary_symmetry(self, parsed: dict, crystal_system: str) -> str:
        """Describe secondary symmetry elements"""
        descriptions = []
        
        for element in parsed['secondary_elements']:
            if element['type'] == 'rotation_axis':
                fold = element['fold']
                descriptions.append(f"{fold}-fold axes in secondary orientations")
                
            elif element['type'] == 'screw_axis':
                fold = element['fold']
                translation = element['translation']
                full_notation = element['full_notation']
                descriptions.append(f"{full_notation} screw axes with {fold}-fold helical symmetry")
        
        return ". ".join(descriptions) if descriptions else ""

    def describe_plane_symmetry(self, parsed: dict) -> str:
        """Describe mirror and glide plane symmetries"""
        descriptions = []
        
        # Group plane elements by type
        mirrors = [p for p in parsed['plane_elements'] if p['type'] == 'mirror']
        glides = [p for p in parsed['plane_elements'] if p['type'] == 'glide']
        
        # Describe mirrors
        if mirrors:
            if len(mirrors) == 1:
                descriptions.append("Contains a mirror plane providing reflection symmetry")
            else:
                descriptions.append(f"Contains {len(mirrors)} mirror planes creating multiple reflection symmetries")
        
        # Describe glides with specific translation information
        if glides:
            glide_types = {}
            for glide in glides:
                glide_type = glide['glide_type']
                if glide_type not in glide_types:
                    glide_types[glide_type] = 0
                glide_types[glide_type] += 1
            
            glide_descriptions = []
            for glide_type, count in glide_types.items():
                if glide_type == 'a':
                    glide_descriptions.append(f"{count} a-glide plane{'s' if count > 1 else ''} with ½a translation")
                elif glide_type == 'b':
                    glide_descriptions.append(f"{count} b-glide plane{'s' if count > 1 else ''} with ½b translation")
                elif glide_type == 'c':
                    glide_descriptions.append(f"{count} c-glide plane{'s' if count > 1 else ''} with ½c translation")
                elif glide_type == 'n':
                    glide_descriptions.append(f"{count} n-glide plane{'s' if count > 1 else ''} with ½(a+b) diagonal translation")
                elif glide_type == 'd':
                    glide_descriptions.append(f"{count} d-glide plane{'s' if count > 1 else ''} with ¼(a+b±c) diamond translation")
                elif glide_type == 'e':
                    glide_descriptions.append(f"{count} e-glide plane{'s' if count > 1 else ''} with alternating ½a, ½b translations")
            
            descriptions.append(f"Features {', '.join(glide_descriptions)}")
        
        return ". ".join(descriptions) if descriptions else ""

    def describe_axis_plane_relationships(self, parsed: dict) -> str:
        """Describe axis/plane combinations when present"""
        if not parsed['has_slash']:
            return ""
        
        axis_part = parsed.get('axis_part', '')
        plane_part = parsed.get('plane_part', '')
        
        if axis_part and plane_part:
            return f"The {axis_part} axis is perpendicular to {plane_part} symmetry elements forming an integrated axis-plane system"
        
        return ""

    def get_systematic_absences_(self, symbol: str, parsed: dict) -> str:
        """Generate systematic absence conditions based on actual symmetry elements"""
        conditions = []
        
        # Lattice centering effects
        centering = symbol[0]
        centering_map = {
            'I': "h+k+l=2n",
            'F': "h,k,l all even or all odd", 
            'A': "k+l=2n",
            'B': "h+l=2n",
            'C': "h+k=2n",
            'R': "-h+k+l=3n"
        }
        if centering in centering_map:
            conditions.append(centering_map[centering])
        
        # Screw axis effects
        all_elements = parsed['primary_elements'] + parsed['secondary_elements']
        for element in all_elements:
            if element['type'] == 'screw_axis':
                fold = element['fold']
                full_notation = element['full_notation']
                conditions.append(f"{full_notation}: l=n×{fold}")
        
        # Glide plane effects
        glide_map = {
            'a': "h=2n", 'b': "k=2n", 'c': "l=2n",
            'n': "h+k=2n", 'd': "complex ¼-cell", 'e': "alternating"
        }
        for plane in parsed['plane_elements']:
            if plane['type'] == 'glide':
                glide_type = plane['glide_type']
                if glide_type in glide_map:
                    conditions.append(f"{glide_type}-glide: {glide_map[glide_type]}")
        
        return "; ".join(conditions) if conditions else "None"

    def generate__fingerprint(self, ita_number: int) -> str:
        """Generate unique fingerprint based on Hermann-Mauguin symbol content"""
        try:
            sg = SpaceGroup.from_int_number(ita_number)
            symbol = sg.symbol
            crystal_system = sg.crystal_system
            point_group = str(sg.point_group)
            
            # Parse Hermann-Mauguin symbol completely
            parsed = self.parse_hermann_mauguin_completely(symbol, crystal_system)
            
            # Generate core descriptions
            lattice_desc = self.get_bravais_lattice_description(symbol)
            primary_desc = self.describe_primary_symmetry(parsed, crystal_system)
            secondary_desc = self.describe_secondary_symmetry(parsed, crystal_system)
            plane_desc = self.describe_plane_symmetry(parsed)
            axis_plane_desc = self.describe_axis_plane_relationships(parsed)
            absence_desc = self.get_systematic_absences_(symbol, parsed)
            
            # Combine symmetry descriptions
            symmetry_parts = [desc for desc in [primary_desc, secondary_desc, plane_desc, axis_plane_desc] if desc]
            symmetry_description = ". ".join(symmetry_parts) if symmetry_parts else "Only lattice translations"
            
            # Build ultra-concise fingerprint
            fingerprint = f"SG {ita_number} ({symbol}): {crystal_system}/{point_group}. {lattice_desc}. {symmetry_description}. Absences: {absence_desc}."
            
            # Or Medium-concise version
            # fingerprint = (
            #     f"Space group {ita_number} ({symbol}): {crystal_system} system, {point_group} point group. "
            #     f"{lattice_desc}. "
            #     f"Symmetry: {symmetry_description}. "
            #     f"Absences: {absence_desc}."
            # )
            
            return fingerprint
            
        except Exception as e:
            return f"Error processing space group {ita_number}: {str(e)}"

    def generate_all__fingerprints(self, output_file: str = 'spacegroup_fingerprints.json') -> dict:
        """Generate all 230 unique space group fingerprints"""
        print("Generating unique fingerprints for all 230 space groups...")
        print("(No artificial templating - purely based on Hermann-Mauguin symbol content)")
        print("=" * 70)
        
        fingerprints = {}
        failed_groups = []
        
        for ita_number in range(1, 231):
            try:
                fingerprint = self.generate__fingerprint(ita_number)
                sg = SpaceGroup.from_int_number(ita_number)
                fingerprints[sg.symbol] = fingerprint
                
                if ita_number % 50 == 0:
                    print(f"✓ Processed {ita_number}/230 space groups...")
                    
            except Exception as e:
                print(f"✗ Error processing space group {ita_number}: {e}")
                failed_groups.append(ita_number)
        
        # Check for uniqueness
        text_counts = defaultdict(list)
        for symbol, text in fingerprints.items():
            text_counts[text].append(symbol)
        
        duplicates = {text: symbols for text, symbols in text_counts.items() if len(symbols) > 1}
        unique_count = len(fingerprints) - sum(len(symbols) - 1 for symbols in duplicates.values())
        
        print(f"\n📊 Uniqueness Results:")
        print(f"   Total fingerprints: {len(fingerprints)}")
        print(f"   unique descriptions: {unique_count}")
        print(f"   duplicate groups: {len(duplicates)}")
        print(f"   uniqueness rate: {(unique_count/len(fingerprints)*100):.1f}%")
        
        if duplicates:
            print(f"\n⚠️  Groups with similar descriptions:")
            for i, (text, symbols) in enumerate(list(duplicates.items())[:10]):
                print(f"   {i+1}. {symbols}")
                if i == 4:  # Show first 5, indicate if more
                    print(f"   ... and {len(duplicates)-5} more groups")
                    break
        
        if failed_groups:
            print(f"\n❌ Failed groups: {failed_groups}")
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fingerprints, f, indent=4, ensure_ascii=False)
        
        print(f"\n💾 Saved to '{output_file}'")
        
        # Calculate detailed sentence statistics
        word_counts = []
        char_counts = []
        for text in fingerprints.values():
            # Count words by splitting on whitespace
            word_count = len(text.split())
            char_count = len(text)
            word_counts.append(word_count)
            char_counts.append(char_count)
        
        if word_counts:
            try:
                import numpy as np
                median_words = np.median(word_counts)
            except ImportError:
                # Fallback if numpy is not available
                sorted_words = sorted(word_counts)
                n = len(sorted_words)
                median_words = sorted_words[n//2] if n % 2 == 1 else (sorted_words[n//2-1] + sorted_words[n//2]) / 2
            
            min_words = min(word_counts)
            max_words = max(word_counts)
            avg_words = sum(word_counts) / len(word_counts)
            
            min_chars = min(char_counts)
            max_chars = max(char_counts)
            avg_chars = sum(char_counts) / len(char_counts)
            
            print(f"\n📝 Sentence Statistics:")
            print(f"   Word count - Min: {min_words}, Max: {max_words}, Avg: {avg_words:.1f}, Median: {median_words:.1f}")
            print(f"   Character count - Min: {min_chars}, Max: {max_chars}, Avg: {avg_chars:.1f}")
            print(f"   Total sentences generated: {len(fingerprints)}")
            
            # Find shortest and longest descriptions
            shortest_idx = word_counts.index(min_words)
            longest_idx = word_counts.index(max_words)
            shortest_symbol = list(fingerprints.keys())[shortest_idx]
            longest_symbol = list(fingerprints.keys())[longest_idx]
            
            print(f"\n📏 Length Examples:")
            print(f"   Shortest ({min_words} words): {shortest_symbol}")
            print(f"   Longest ({max_words} words): {longest_symbol}")
        else:
            print(f"\n📝 Sentence Statistics:")
            print(f"   No valid sentences generated")
        
        print(f"\n🔍 Analysis: uniqueness is {unique_count}/230 = {(unique_count/230*100):.1f}%")
        if unique_count < 230:
            print(f"   The remaining {230-unique_count} duplicates represent space groups with")
            print(f"   genuinely similar symmetry characteristics that may require")
            print(f"   additional distinguishing features beyond basic Hermann-Mauguin parsing.")
        
        # Save statistics to text file
        stats_file = output_file.replace('.json', '_statistics.txt')
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("Space Group Fingerprint Generation Statistics\n")
                f.write("=" * 50 + "\n\n")
                
                # Generation results
                f.write("Generation Results:\n")
                f.write(f"   Total fingerprints: {len(fingerprints)}\n")
                f.write(f"   Unique descriptions: {unique_count}\n")
                f.write(f"   Duplicate groups: {len(duplicates)}\n")
                f.write(f"   Uniqueness rate: {(unique_count/len(fingerprints)*100):.1f}%\n\n")
                
                # Sentence statistics
                if word_counts:
                    f.write("Sentence Statistics:\n")
                    f.write(f"   Word count - Min: {min_words}, Max: {max_words}, Avg: {avg_words:.1f}, Median: {median_words:.1f}\n")
                    f.write(f"   Character count - Min: {min_chars}, Max: {max_chars}, Avg: {avg_chars:.1f}\n")
                    f.write(f"   Total sentences generated: {len(fingerprints)}\n\n")
                    
                    f.write("Length Examples:\n")
                    f.write(f"   Shortest ({min_words} words): {shortest_symbol}\n")
                    f.write(f"   Longest ({max_words} words): {longest_symbol}\n\n")
                
                # Duplicate groups details
                if duplicates:
                    f.write("Groups with Similar Descriptions:\n")
                    for i, (text, symbols) in enumerate(duplicates.items()):
                        f.write(f"   {i+1}. {symbols}\n")
                        if len(text) > 100:
                            f.write(f"      Description: {text[:100]}...\n")
                        else:
                            f.write(f"      Description: {text}\n")
                        f.write("\n")
                else:
                    f.write("No duplicate descriptions found - all 230 space groups have unique fingerprints!\n\n")
                
                # Failed groups
                if failed_groups:
                    f.write("Failed Groups:\n")
                    f.write(f"   {failed_groups}\n\n")
                else:
                    f.write("All space groups processed successfully!\n\n")
                
                # Analysis summary
                f.write("Analysis Summary:\n")
                f.write(f"   Uniqueness achieved: {unique_count}/230 = {(unique_count/230*100):.1f}%\n")
                if unique_count < 230:
                    f.write(f"   The remaining {230-unique_count} duplicates represent space groups with\n")
                    f.write(f"   genuinely similar symmetry characteristics that may require\n")
                    f.write(f"   additional distinguishing features beyond basic Hermann-Mauguin parsing.\n")
                else:
                    f.write("   Perfect uniqueness achieved through Hermann-Mauguin symbol parsing!\n")
            
            print(f"\n📄 Statistics saved to '{stats_file}'")
            
        except Exception as e:
            print(f"\n⚠️  Failed to save statistics file: {e}")
        
        return fingerprints

# Test and execution
def main():
    # Generate all fingerprints
    descriptor = SpaceGroupDescriptor()
    all_fingerprints = descriptor.generate_all__fingerprints()
    
    return all_fingerprints

if __name__ == "__main__":
    fingerprints = main()