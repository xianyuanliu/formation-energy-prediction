# Natural Unique Space Group Description Generator
# Relies on inherent uniqueness of Hermann-Mauguin notation without artificial templating

import json
from pymatgen.symmetry.groups import SpaceGroup
import re
from collections import defaultdict

class NaturalSpaceGroupDescriptor:
    def __init__(self):
        # Comprehensive symmetry element analysis patterns
        self.screw_pattern = r'(\d+)_(\d+)'
        self.glide_chars = set('abcdne')
        
    def get_bravais_lattice_description(self, symbol: str) -> str:
        """Bravais lattice description - your working version"""
        first_char = symbol[0]
        desc_map = {
            'P': 'a **Primitive (P)** Bravais lattice, which means lattice points are located only at the corners of the unit cell',
            'I': 'a **Body-centered (I)** Bravais lattice, with lattice points at the corners and at the center of the unit cell',
            'F': 'a **Face-centered (F)** Bravais lattice, with lattice points at the corners and at the center of each of the six faces',
            'A': 'an **A-face-centered (A)** Bravais lattice, with lattice points at the corners and at the center of the A faces',
            'B': 'a **B-face-centered (B)** Bravais lattice, with lattice points at the corners and at the center of the B faces',
            'C': 'a **C-face-centered (C)** Bravais lattice, with lattice points at the corners and at the center of the C faces',
            'R': 'a **Rhombohedral (R)** Bravais lattice with a primitive hexagonal cell or rhombohedral setting'
        }
        return desc_map.get(first_char, 'an unknown Bravais lattice type')

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
                        descriptions.append("Contains **4-fold rotation axes** along cubic <100> directions")
                    elif fold == '3':
                        descriptions.append("Features **3-fold rotation axes** along <111> body diagonals")
                    elif fold == '2':
                        descriptions.append("Exhibits **2-fold rotation axes** in cubic symmetry")
                        
                elif crystal_system == 'tetragonal':
                    if fold == '4':
                        descriptions.append("Possesses a **principal 4-fold rotation axis** along the c-direction")
                    elif fold == '2':
                        descriptions.append("Contains **2-fold rotation axes** perpendicular to the principal axis")
                        
                elif crystal_system == 'hexagonal':
                    if fold == '6':
                        descriptions.append("Contains a **6-fold rotation axis** defining hexagonal symmetry")
                    elif fold == '3':
                        descriptions.append("Features a **3-fold rotation axis** as the primary element")
                    elif fold == '2':
                        descriptions.append("Includes **2-fold rotation axes** in the hexagonal framework")
                        
                elif crystal_system == 'trigonal':
                    if fold == '3':
                        if 'R' in parsed['symbol']:
                            descriptions.append("Exhibits **3-fold rotation** in rhombohedral setting")
                        else:
                            descriptions.append("Contains **3-fold trigonal rotation** as the defining element")
                            
                elif crystal_system == 'orthorhombic':
                    descriptions.append(f"Contains **{fold}-fold rotation** along a principal crystallographic axis")
                    
                elif crystal_system == 'monoclinic':
                    descriptions.append(f"Features a **unique {fold}-fold rotation axis** as the only rotational symmetry")
                    
            elif element['type'] == 'screw_axis':
                fold = element['fold']
                translation = element['translation']
                full_notation = element['full_notation']
                
                # Calculate the actual translation fraction
                trans_fraction = f"{translation}/{fold}"
                
                descriptions.append(f"Incorporates a **{full_notation} screw axis** combining {fold}-fold rotation with {trans_fraction} unit cell translation")
                
            elif element['type'] == 'rotoinversion_axis':
                fold = element['fold']
                full_notation = element['full_notation']
                descriptions.append(f"Features a **{full_notation} rotoinversion axis** combining {fold}-fold rotation with inversion")
        
        return ". ".join(descriptions) if descriptions else ""

    def describe_secondary_symmetry(self, parsed: dict, crystal_system: str) -> str:
        """Describe secondary symmetry elements"""
        descriptions = []
        
        for element in parsed['secondary_elements']:
            if element['type'] == 'rotation_axis':
                fold = element['fold']
                descriptions.append(f"**{fold}-fold axes** in secondary orientations")
                
            elif element['type'] == 'screw_axis':
                fold = element['fold']
                translation = element['translation']
                full_notation = element['full_notation']
                descriptions.append(f"**{full_notation} screw axes** with {fold}-fold helical symmetry")
        
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
                descriptions.append("Contains a **mirror plane** providing reflection symmetry")
            else:
                descriptions.append(f"Contains **{len(mirrors)} mirror planes** creating multiple reflection symmetries")
        
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
                    glide_descriptions.append(f"**{count} a-glide plane{'s' if count > 1 else ''}** with ½a translation")
                elif glide_type == 'b':
                    glide_descriptions.append(f"**{count} b-glide plane{'s' if count > 1 else ''}** with ½b translation")
                elif glide_type == 'c':
                    glide_descriptions.append(f"**{count} c-glide plane{'s' if count > 1 else ''}** with ½c translation")
                elif glide_type == 'n':
                    glide_descriptions.append(f"**{count} n-glide plane{'s' if count > 1 else ''}** with ½(a+b) diagonal translation")
                elif glide_type == 'd':
                    glide_descriptions.append(f"**{count} d-glide plane{'s' if count > 1 else ''}** with ¼(a+b±c) diamond translation")
                elif glide_type == 'e':
                    glide_descriptions.append(f"**{count} e-glide plane{'s' if count > 1 else ''}** with alternating ½a, ½b translations")
            
            descriptions.append(f"Features {', '.join(glide_descriptions)}")
        
        return ". ".join(descriptions) if descriptions else ""

    def describe_axis_plane_relationships(self, parsed: dict) -> str:
        """Describe axis/plane combinations when present"""
        if not parsed['has_slash']:
            return ""
        
        axis_part = parsed.get('axis_part', '')
        plane_part = parsed.get('plane_part', '')
        
        if axis_part and plane_part:
            return f"The **{axis_part} axis is perpendicular to {plane_part} symmetry elements** forming an integrated axis-plane system"
        
        return ""

    def get_systematic_absences_natural(self, symbol: str, parsed: dict) -> str:
        """Generate systematic absence conditions based on actual symmetry elements"""
        conditions = []
        
        # Lattice centering effects
        centering = symbol[0]
        if centering == 'I':
            conditions.append("body-centering requires h+k+l=2n for allowed reflections")
        elif centering == 'F':
            conditions.append("face-centering demands h,k,l all even or all odd")
        elif centering == 'A':
            conditions.append("A-face-centering enforces k+l=2n reflection conditions")
        elif centering == 'B':
            conditions.append("B-face-centering creates h+l=2n systematic rules")
        elif centering == 'C':
            conditions.append("C-face-centering produces h+k=2n extinction conditions")
        elif centering == 'R':
            conditions.append("rhombohedral centering introduces -h+k+l=3n reflection requirements")
        
        # Screw axis effects
        all_elements = parsed['primary_elements'] + parsed['secondary_elements']
        for element in all_elements:
            if element['type'] == 'screw_axis':
                fold = element['fold']
                translation = element['translation']
                full_notation = element['full_notation']
                conditions.append(f"{full_notation} screw axis creates l=n×{fold} systematic extinctions")
        
        # Glide plane effects
        for plane in parsed['plane_elements']:
            if plane['type'] == 'glide':
                glide_type = plane['glide_type']
                if glide_type == 'a':
                    conditions.append("a-glide planes require h=2n for allowed reflections")
                elif glide_type == 'b':
                    conditions.append("b-glide planes enforce k=2n systematic conditions")
                elif glide_type == 'c':
                    conditions.append("c-glide planes create l=2n extinction rules")
                elif glide_type == 'n':
                    conditions.append("n-glide planes produce h+k=2n systematic absences")
                elif glide_type == 'd':
                    conditions.append("d-glide planes generate complex ¼-cell translation extinctions")
                elif glide_type == 'e':
                    conditions.append("e-glide planes create alternating translation systematic rules")
        
        return "; ".join(conditions) if conditions else "No systematic absences - all reflections are crystallographically allowed"

    def generate_natural_fingerprint(self, ita_number: int) -> str:
        """Generate naturally unique fingerprint based on Hermann-Mauguin symbol content"""
        try:
            sg = SpaceGroup.from_int_number(ita_number)
            symbol = sg.symbol
            crystal_system = sg.crystal_system
            point_group = str(sg.point_group)
            
            # Parse Hermann-Mauguin symbol completely
            parsed = self.parse_hermann_mauguin_completely(symbol, crystal_system)
            
            # Generate descriptions based on actual content
            lattice_desc = self.get_bravais_lattice_description(symbol)
            primary_desc = self.describe_primary_symmetry(parsed, crystal_system)
            secondary_desc = self.describe_secondary_symmetry(parsed, crystal_system)
            plane_desc = self.describe_plane_symmetry(parsed)
            axis_plane_desc = self.describe_axis_plane_relationships(parsed)
            absence_desc = self.get_systematic_absences_natural(symbol, parsed)
            
            # Combine all descriptions naturally
            symmetry_parts = [desc for desc in [primary_desc, secondary_desc, plane_desc, axis_plane_desc] if desc]
            symmetry_description = ". ".join(symmetry_parts) if symmetry_parts else "Contains only fundamental lattice translation symmetry"
            
            # Get multiplicity
            multiplicity = len(sg.symmetry_ops)
            
            # Build final fingerprint
            fingerprint = (
                f"Space group {ita_number} with Hermann-Mauguin symbol {symbol} belongs to the {crystal_system} "
                f"crystal system and {point_group} point group. It utilizes {lattice_desc}. "
                f"Key symmetry elements: {symmetry_description}. "
                f"Systematic absence conditions: {absence_desc}. "
                f"General position multiplicity: {multiplicity}."
            )
            
            return fingerprint
            
        except Exception as e:
            return f"Error processing space group {ita_number}: {str(e)}"

    def generate_all_natural_fingerprints(self, output_file: str = 'spacegroup_fingerprints.json') -> dict:
        """Generate all 230 naturally unique space group fingerprints"""
        print("Generating naturally unique fingerprints for all 230 space groups...")
        print("(No artificial templating - purely based on Hermann-Mauguin symbol content)")
        print("=" * 70)
        
        fingerprints = {}
        failed_groups = []
        
        for ita_number in range(1, 231):
            try:
                fingerprint = self.generate_natural_fingerprint(ita_number)
                sg = SpaceGroup.from_int_number(ita_number)
                fingerprints[sg.symbol] = fingerprint
                
                if ita_number % 50 == 0:
                    print(f"✓ Processed {ita_number}/230 space groups...")
                    
            except Exception as e:
                print(f"✗ Error processing space group {ita_number}: {e}")
                failed_groups.append(ita_number)
        
        # Check for natural uniqueness
        text_counts = defaultdict(list)
        for symbol, text in fingerprints.items():
            text_counts[text].append(symbol)
        
        duplicates = {text: symbols for text, symbols in text_counts.items() if len(symbols) > 1}
        unique_count = len(fingerprints) - sum(len(symbols) - 1 for symbols in duplicates.values())
        
        print(f"\n📊 Natural Uniqueness Results:")
        print(f"   Total fingerprints: {len(fingerprints)}")
        print(f"   Naturally unique descriptions: {unique_count}")
        print(f"   Naturally duplicate groups: {len(duplicates)}")
        print(f"   Natural uniqueness rate: {(unique_count/len(fingerprints)*100):.1f}%")
        
        if duplicates:
            print(f"\n⚠️  Groups with naturally similar descriptions:")
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
        print(f"\n🔍 Analysis: Natural uniqueness is {unique_count}/230 = {(unique_count/230*100):.1f}%")
        if unique_count < 230:
            print(f"   The remaining {230-unique_count} duplicates represent space groups with")
            print(f"   genuinely similar symmetry characteristics that may require")
            print(f"   additional distinguishing features beyond basic Hermann-Mauguin parsing.")
        
        return fingerprints

# Test and execution
def main():
    """Test natural uniqueness approach"""
    descriptor = NaturalSpaceGroupDescriptor()
    
    # Test with a few examples
    print("Testing natural description generation:")
    print("-" * 50)
    
    test_groups = [1, 2, 14, 62, 123, 225]  # Diverse examples
    for sg_num in test_groups:
        fingerprint = descriptor.generate_natural_fingerprint(sg_num)
        sg = SpaceGroup.from_int_number(sg_num)
        print(f"\n{sg.symbol} (#{sg_num}):")
        print(f"{fingerprint}")
        print("-" * 50)
    
    # Generate all fingerprints
    print("\n" + "="*70)
    all_fingerprints = descriptor.generate_all_natural_fingerprints()
    
    return all_fingerprints

if __name__ == "__main__":
    fingerprints = main()