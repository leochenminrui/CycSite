"""
CycPeptMPDB (PAMPA) to PEM schema converter.

Dataset-specific parsing logic for CycPeptMPDB PAMPA permeability data.

Handles cyclic peptides with custom monomer notation including:
- N-methylated amino acids (meL, meA, etc.)
- D-amino acids (dL, dA, etc.)
- N-methylated D-amino acids (Me_dL, Me_dA, etc.)
- Non-standard amino acids (Abu, Sar, Nle, etc.)
- N-terminal modifications (ac-, etc.)
- C-terminal modifications (-pip, etc.)
- Cyclization (Circle, Lariat)
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import ast
import pandas as pd

from .base_converter import BaseConverter
from ..pem_schema import Edit, EditFamily, AnchorKind, ParserStatus


class CycPeptMPDBConverter(BaseConverter):
    """
    Converter for CycPeptMPDB PAMPA dataset.

    Handles cyclic peptides with membrane permeability measurements.
    """

    def __init__(self, strict_mode: bool = True):
        super().__init__(
            dataset_name="CycPeptMPDB_PAMPA",
            parser_version="1.0.0",
            strict_mode=strict_mode
        )

        # Standard amino acid alphabet
        self.standard_aa = set('ACDEFGHIKLMNPQRSTVWY')

        # Monomer to standard AA mapping
        self.monomer_to_aa = self._build_monomer_mapping()

        # Modification patterns
        self.n_methyl_pattern = re.compile(r'^me[A-Z]$')  # meL, meA, etc.
        self.d_aa_pattern = re.compile(r'^d[A-Z]$')  # dL, dA, etc.
        self.n_methyl_d_aa_pattern = re.compile(r'^Me_d[A-Z]$')  # Me_dL, Me_dA, etc.

    def _build_monomer_mapping(self) -> Dict[str, str]:
        """
        Build mapping from monomer codes to standard amino acids.

        Returns:
            Dictionary mapping monomer codes to single-letter AA codes
        """
        mapping = {}

        # Standard AA (already single letter)
        for aa in self.standard_aa:
            mapping[aa] = aa

        # N-methylated amino acids (meL -> L)
        for aa in self.standard_aa:
            mapping[f'me{aa}'] = aa

        # D-amino acids (dL -> L)
        for aa in self.standard_aa:
            mapping[f'd{aa}'] = aa

        # N-methylated D-amino acids (Me_dL -> L)
        for aa in self.standard_aa:
            mapping[f'Me_d{aa}'] = aa

        # Non-standard amino acids - map to similar standard AA
        mapping.update({
            'Abu': 'A',  # α-aminobutyric acid -> Ala
            'Sar': 'G',  # Sarcosine (N-methylglycine) -> Gly
            'Nle': 'L',  # Norleucine -> Leu
            'Nva': 'V',  # Norvaline -> Val
            'Orn': 'K',  # Ornithine -> Lys
            'Cha': 'F',  # Cyclohexylalanine -> Phe
            'Hph': 'F',  # Homophenylalanine -> Phe
            'bHph': 'F',  # beta-homophenylalanine -> Phe
            'Nal': 'F',  # Naphthylalanine -> Phe
            '1-Nal': 'F',
            'd1-Nal': 'F',
            'Phg': 'F',  # Phenylglycine -> Phe
            'Phe(4-F)': 'F',
            'Phe(4-CF3)': 'F',
            'Phe(4-NO2)': 'F',
            'dPhe(4-F)': 'F',
            'dPhe(3,4-diF)': 'F',
            'Et_Phe': 'F',
            'H2NEt_Phe': 'F',
            'Me_Phe(3-Cl)': 'F',
            'Phe(O->S)': 'F',
            'Tyr(Me)': 'Y',
            'Tyr(3-F)': 'Y',
            'Tyr(O->S)': 'Y',
            'Me_Tyr(Me)': 'Y',
            'dY': 'Y',
            'dTyr(bR-OMe)': 'Y',
            'Trp(5-Br)': 'W',
            'Trp(6-Br)': 'W',
            'Trp(7-Br)': 'W',
            'dW': 'W',
            'Me_dW': 'W',
            'dTrp(O->S)': 'W',
            'Ser(tBu)': 'S',
            'Ser(Me)': 'S',
            'Ser(Ac)': 'S',
            'dS': 'S',
            'dSer(Me)': 'S',
            'Thr(O->S)': 'T',
            'dT': 'T',
            'Ile(O->S)': 'I',
            'dI': 'I',
            'Me_dI': 'I',
            'meI': 'I',
            'xiIle': 'I',
            'd(N->O)aIle': 'I',
            'Leu(O->S)': 'L',
            'dLeu(3R-OH)': 'L',
            'dLeu(O->S)': 'L',
            'aMeLeu': 'L',
            'd(N->O)Leu': 'L',
            '(N->O)Leu': 'L',
            'Val(O->S)': 'V',
            'dV': 'V',
            'Me_dV': 'V',
            'meV': 'V',
            'd(N->O)Val': 'V',
            '(N->O)Val': 'V',
            '(N->O)Val(3-OH)': 'V',
            'Lys(Ac)': 'K',
            'Lys(Me2)': 'K',
            'Lys(Tfa)': 'K',
            'Lys(O->S)': 'K',
            'dK': 'K',
            'Me_dK': 'K',
            'meK': 'K',
            'Arg(Me,Me)': 'R',
            'dR': 'R',
            'Asp(OMe)': 'D',
            'Asp(Ph(2-NH2))': 'D',
            'dAsp(pyrrol-1-yl)': 'D',
            'Glu(3R-Me)': 'E',
            'Glu(OMe)': 'E',
            'dGlu(OMe)': 'E',
            'Gln(Me)': 'Q',
            'Gln(Me2)': 'Q',
            'dGln(Me2)': 'Q',
            'meQ': 'Q',
            'Asn(Me2)': 'N',
            'dAsn(Me2)': 'N',
            'meN': 'N',
            'Met(O2)': 'M',
            'meM': 'M',
            'Pro(O->S)': 'P',
            'dP': 'P',
            'dPro(O->S)': 'P',
            'dHyp': 'P',
            'Pip': 'P',
            'Pye': 'P',
            'dPye': 'P',
            'Aib': 'A',  # α-aminoisobutyric acid
            'meA': 'A',
            'Me_dA': 'A',
            'dA': 'A',
            'dAbu': 'A',
            'Me_dAbu': 'A',
            'dAla(O->S)': 'A',
            'Ala(O->S)': 'A',
            'Ala(5-Tet)': 'A',
            'Ala(indol-2-yl)': 'A',
            'Me_Ala(indol-2-yl)': 'A',
            'dAla(indol-2-yl)': 'A',
            'Gly': 'G',
            'Bn_Gly': 'G',
            'Pr_Gly': 'G',
            'Et_Gly': 'G',
            'MeOEt_Gly': 'G',
            'iBu_Gly': 'G',
            'NH2Bu_Gly': 'G',
            'cHexCH2_Gly': 'G',
            'PhPr_Gly': 'G',
            '3-pyridylethyl_Gly': 'G',
            'd(N->O)Gly(allyl)': 'G',
            'Tle': 'L',  # tert-leucine
            'Bal': 'A',  # β-alanine
            'Bal(3-Me)': 'A',
            'Bal(d3-CF3)': 'A',
            'HOCOCH2_Bal': 'A',
            'Me_Bal': 'A',
            'Sta': 'L',  # Statine
            'Sta(3R,4R)': 'L',
            'Chg': 'L',  # Cyclohexylglycine
            'Me_Cha': 'F',
            'dCha': 'F',
            'Me_Nle': 'L',
            'Me_dNle': 'L',
            'dNle': 'L',
            'Me_Nva': 'V',
            'Me_dNva': 'V',
            'dNva': 'V',
            'Nva(Ph)': 'V',
            'meF': 'F',
            'Me_dF': 'F',
            'dF': 'F',
            'meS': 'S',
            'meT': 'T',
            'meW': 'W',
            'meY': 'Y',

            # ── 17 monomers added after full 7298-row audit (2026-06-16) ──
            'pentyl_Gly': 'G',       # pentyl-glycine → Gly backbone
            'Bn(4-Cl)_Gly': 'G',     # 4-chlorobenzyl-glycine → Gly backbone
            'GABA': 'G',             # γ-aminobutyric acid → Gly backbone
            'Me_Abu': 'A',           # N-methyl-aminobutyric acid → Ala backbone
            'Me_Bmt(E)': 'T',        # N-methyl-butenyl-methyl-threonine → Thr backbone
            'Aoc(2)': 'L',           # 2-aminooctanoic acid → Leu backbone
            '5-Ava': 'G',            # 5-aminovaleric acid → Gly backbone
            'Cys(EtO2H)_NH2': 'C',   # cysteine ethyl ester → Cys backbone
            '3Pal': 'F',             # 3-pyridylalanine → Phe backbone
            'HOCOCH2_Gly_ol': 'G',   # glycolic acid derivative → Gly backbone
            'Glu_NH2': 'Q',          # glutamine (amidated Glu) → Gln backbone
            'Tza': 'P',              # thiazolidine → Pro backbone (ring surrogate)
            'Asp_piperidide': 'D',   # aspartic acid piperidide → Asp backbone
            'Me_Phe(a,b-dehydro)': 'F',  # N-methyl dehydro-Phe → Phe backbone
            '2Abz': 'F',             # 2-aminobenzoic acid → Phe backbone
            '(N->O)xiIle': 'I',      # N→O isostere of Ile → Ile backbone
            '(N->O)Tyr': 'Y',        # N→O isostere of Tyr → Tyr backbone
        })

        return mapping

    def _parse_monomer_list(self, seq_str: str) -> Optional[List[str]]:
        """
        Parse the Sequence column which contains Python list notation.

        Args:
            seq_str: String representation of list, e.g. "['L', 'meL', 'dP']"

        Returns:
            List of monomer codes or None if parsing failed
        """
        if not seq_str or pd.isna(seq_str):
            return None

        try:
            # Use ast.literal_eval for safe parsing
            monomer_list = ast.literal_eval(str(seq_str))
            if isinstance(monomer_list, list):
                return monomer_list
        except (ValueError, SyntaxError):
            pass

        return None

    def parse_sequence(self, raw_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract canonical backbone sequence from monomer list.

        Args:
            raw_data: Raw row data

        Returns:
            Canonical sequence (standard amino acids only) or None
        """
        # Get the Sequence column
        seq_str = raw_data.get('Sequence')
        if not seq_str:
            return None

        # Parse monomer list
        monomers = self._parse_monomer_list(seq_str)
        if not monomers:
            return None

        # Convert monomers to backbone sequence
        backbone = []
        for monomer in monomers:
            # Skip terminal modifications
            if monomer.startswith('ac-') or monomer.startswith('-') or monomer.endswith('-'):
                continue

            # Skip 'Mono' placeholders (unknown structures)
            if monomer.startswith('Mono'):
                continue

            # Skip complex modifiers
            if monomer in ['glyco-', 'deca-', 'medl-']:
                continue

            # Map to standard AA
            std_aa = self.monomer_to_aa.get(monomer)
            if std_aa:
                backbone.append(std_aa)
            else:
                # Unknown monomer - skip but track for debugging
                if self.strict_mode:
                    # In strict mode, require all monomers to be mappable
                    return None
                # In relaxed mode, skip unknown monomers
                continue

        if len(backbone) >= 3:
            return ''.join(backbone)
        else:
            return None

    def parse_edits(
        self,
        raw_data: Dict[str, Any],
        sequence: str,
        sample_id: str
    ) -> List[Edit]:
        """
        Parse chemical modifications from monomer list.

        Identifies:
        - N-methylation (meL, meA, etc.)
        - D-amino acids (dL, dA, etc.)
        - N-methylated D-amino acids (Me_dL, etc.)
        - N-terminal modifications (ac-)
        - C-terminal modifications (-pip, etc.)
        - Cyclization (from Molecule_Shape)

        Args:
            raw_data: Raw row data
            sequence: Canonical sequence
            sample_id: Sample identifier

        Returns:
            List of Edit objects
        """
        edits = []
        edit_counter = 0

        # Parse monomer list
        seq_str = raw_data.get('Sequence')
        monomers = self._parse_monomer_list(seq_str) if seq_str else None
        if not monomers:
            return edits

        # Track position in backbone sequence
        seq_pos = 0

        # Process each monomer
        for monomer_idx, monomer in enumerate(monomers):

            # Check for N-terminal acetylation
            if monomer == 'ac-' and monomer_idx == 0:
                edit = Edit(
                    edit_id=f"{sample_id}_edit_{edit_counter}",
                    edit_family=EditFamily.N_TERMINAL,
                    edit_type="n_acetylation",
                    anchor_kind=AnchorKind.EXPLICIT,
                    anchor_positions=[0],
                    anchor_residues=[sequence[0]] if len(sequence) > 0 else [],
                    chem_rep_raw="ac-",
                    chem_rep_canonical="Ac-",
                    attachment_semantics="n_terminus",
                    parser_status=ParserStatus.SUCCESS,
                    rule_id="cycpept_n_acetyl_v1",
                    edit_metadata={}
                )
                edits.append(edit)
                edit_counter += 1
                continue

            # Check for C-terminal modifications
            if monomer.startswith('-'):
                c_term_pos = len(sequence) - 1
                edit = Edit(
                    edit_id=f"{sample_id}_edit_{edit_counter}",
                    edit_family=EditFamily.C_TERMINAL,
                    edit_type="c_terminal_modification",
                    anchor_kind=AnchorKind.EXPLICIT,
                    anchor_positions=[c_term_pos],
                    anchor_residues=[sequence[c_term_pos]] if c_term_pos >= 0 else [],
                    chem_rep_raw=monomer,
                    chem_rep_canonical=monomer,
                    attachment_semantics="c_terminus",
                    parser_status=ParserStatus.SUCCESS,
                    rule_id="cycpept_c_term_v1",
                    edit_metadata={"c_term_group": monomer}
                )
                edits.append(edit)
                edit_counter += 1
                continue

            # Skip terminal markers and unknowns
            if (monomer.startswith('Mono') or monomer.endswith('-') or
                monomer in ['glyco-', 'deca-', 'medl-']):
                continue

            # Get the backbone AA for this monomer
            std_aa = self.monomer_to_aa.get(monomer)
            if not std_aa:
                continue

            # Check for N-methylated D-amino acid (Me_dL, Me_dA, etc.)
            if self.n_methyl_d_aa_pattern.match(monomer):
                aa_letter = monomer[-1]  # Last character is the AA
                if seq_pos < len(sequence) and sequence[seq_pos] == aa_letter:
                    # Add D-amino acid modification
                    edit = Edit(
                        edit_id=f"{sample_id}_edit_{edit_counter}",
                        edit_family=EditFamily.BACKBONE,
                        edit_type="d_amino_acid",
                        anchor_kind=AnchorKind.EXPLICIT,
                        anchor_positions=[seq_pos],
                        anchor_residues=[aa_letter],
                        chem_rep_raw=monomer,
                        chem_rep_canonical=f"D-{aa_letter}",
                        attachment_semantics="backbone_stereochemistry",
                        parser_status=ParserStatus.SUCCESS,
                        rule_id="cycpept_d_aa_v1",
                        edit_metadata={"stereochemistry": "D"}
                    )
                    edits.append(edit)
                    edit_counter += 1

                    # Add N-methylation modification
                    edit = Edit(
                        edit_id=f"{sample_id}_edit_{edit_counter}",
                        edit_family=EditFamily.BACKBONE,
                        edit_type="n_methylation",
                        anchor_kind=AnchorKind.EXPLICIT,
                        anchor_positions=[seq_pos],
                        anchor_residues=[aa_letter],
                        chem_rep_raw=monomer,
                        chem_rep_canonical=f"N-Me-{aa_letter}",
                        attachment_semantics="backbone_nitrogen",
                        parser_status=ParserStatus.SUCCESS,
                        rule_id="cycpept_n_methyl_v1",
                        edit_metadata={}
                    )
                    edits.append(edit)
                    edit_counter += 1

            # Check for D-amino acid (dL, dA, etc.)
            elif self.d_aa_pattern.match(monomer):
                aa_letter = monomer[1]  # Second character is the AA
                if seq_pos < len(sequence) and sequence[seq_pos] == aa_letter:
                    edit = Edit(
                        edit_id=f"{sample_id}_edit_{edit_counter}",
                        edit_family=EditFamily.BACKBONE,
                        edit_type="d_amino_acid",
                        anchor_kind=AnchorKind.EXPLICIT,
                        anchor_positions=[seq_pos],
                        anchor_residues=[aa_letter],
                        chem_rep_raw=monomer,
                        chem_rep_canonical=f"D-{aa_letter}",
                        attachment_semantics="backbone_stereochemistry",
                        parser_status=ParserStatus.SUCCESS,
                        rule_id="cycpept_d_aa_v1",
                        edit_metadata={"stereochemistry": "D"}
                    )
                    edits.append(edit)
                    edit_counter += 1

            # Check for N-methylation (meL, meA, etc.)
            elif self.n_methyl_pattern.match(monomer):
                aa_letter = monomer[2]  # Third character is the AA
                if seq_pos < len(sequence) and sequence[seq_pos] == aa_letter:
                    edit = Edit(
                        edit_id=f"{sample_id}_edit_{edit_counter}",
                        edit_family=EditFamily.BACKBONE,
                        edit_type="n_methylation",
                        anchor_kind=AnchorKind.EXPLICIT,
                        anchor_positions=[seq_pos],
                        anchor_residues=[aa_letter],
                        chem_rep_raw=monomer,
                        chem_rep_canonical=f"N-Me-{aa_letter}",
                        attachment_semantics="backbone_nitrogen",
                        parser_status=ParserStatus.SUCCESS,
                        rule_id="cycpept_n_methyl_v1",
                        edit_metadata={}
                    )
                    edits.append(edit)
                    edit_counter += 1

            # Check for non-standard amino acid substitution
            elif monomer not in self.standard_aa and std_aa in self.standard_aa:
                if seq_pos < len(sequence) and sequence[seq_pos] == std_aa:
                    edit = Edit(
                        edit_id=f"{sample_id}_edit_{edit_counter}",
                        edit_family=EditFamily.SUBSTITUTION,
                        edit_type="non_standard_aa",
                        anchor_kind=AnchorKind.EXPLICIT,
                        anchor_positions=[seq_pos],
                        anchor_residues=[std_aa],
                        chem_rep_raw=monomer,
                        chem_rep_canonical=monomer,
                        attachment_semantics="residue_replacement",
                        parser_status=ParserStatus.SUCCESS,
                        rule_id="cycpept_nonstandard_v1",
                        edit_metadata={"original_monomer": monomer, "mapped_to": std_aa}
                    )
                    edits.append(edit)
                    edit_counter += 1

            # Increment sequence position
            seq_pos += 1

        # Check for cyclization from Molecule_Shape
        molecule_shape = raw_data.get('Molecule_Shape')
        if molecule_shape and not pd.isna(molecule_shape):
            shape_str = str(molecule_shape).strip()

            if shape_str == 'Circle':
                # Head-to-tail cyclization
                edit = Edit(
                    edit_id=f"{sample_id}_edit_{edit_counter}",
                    edit_family=EditFamily.CYCLIZATION,
                    edit_type="head_to_tail_cyclization",
                    anchor_kind=AnchorKind.EXPLICIT,
                    anchor_positions=[0, len(sequence) - 1],
                    anchor_residues=[sequence[0], sequence[-1]] if len(sequence) > 0 else [],
                    chem_rep_raw="Circle",
                    chem_rep_canonical=f"cyclic(1-{len(sequence)})",
                    attachment_semantics="backbone_cyclization",
                    parser_status=ParserStatus.SUCCESS,
                    rule_id="cycpept_circle_v1",
                    edit_metadata={"cyclization_type": "head_to_tail", "ring_size": len(sequence)}
                )
                edits.append(edit)
                edit_counter += 1

            elif shape_str == 'Lariat':
                # Lariat structure - branch cyclization
                edit = Edit(
                    edit_id=f"{sample_id}_edit_{edit_counter}",
                    edit_family=EditFamily.CYCLIZATION,
                    edit_type="lariat_cyclization",
                    anchor_kind=AnchorKind.INFERRED,
                    anchor_positions=[],  # Ambiguous - would need SMILES to determine
                    anchor_residues=[],
                    chem_rep_raw="Lariat",
                    chem_rep_canonical="lariat",
                    attachment_semantics="branch_cyclization",
                    parser_status=ParserStatus.INFERRED,
                    rule_id="cycpept_lariat_v1",
                    edit_metadata={"cyclization_type": "lariat"}
                )
                edits.append(edit)
                edit_counter += 1

        return edits

    def parse_label(self, raw_data: Dict[str, Any]) -> Tuple[float, str, str]:
        """
        Extract PAMPA permeability label.

        Args:
            raw_data: Raw row data

        Returns:
            (label_value, label_type, label_unit)
        """
        # Get PAMPA value
        pampa_value = raw_data.get('PAMPA')

        if pampa_value is None or pd.isna(pampa_value):
            raise ValueError("PAMPA value is missing or null")

        try:
            label_value = float(pampa_value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert PAMPA value to float: {pampa_value}")

        # PAMPA values are log permeability
        label_type = "log_permeability"
        label_unit = "log(cm/s)"

        return label_value, label_type, label_unit

    def parse_assay_metadata(self, raw_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Extract PAMPA assay metadata.

        Args:
            raw_data: Raw row data

        Returns:
            (assay_type, assay_metadata)
        """
        assay_type = "PAMPA"

        # Extract available metadata
        assay_metadata = {
            "assay_method": "PAMPA",
            "membrane_type": "artificial",
            "measurement_type": "permeability"
        }

        # Add source information
        source = raw_data.get('Source')
        if source and not pd.isna(source):
            assay_metadata['source_publication'] = str(source)

        year = raw_data.get('Year')
        if year and not pd.isna(year):
            try:
                assay_metadata['publication_year'] = int(year)
            except:
                pass

        # Add SMILES if available
        smiles = raw_data.get('SMILES')
        if smiles and not pd.isna(smiles):
            assay_metadata['smiles'] = str(smiles)

        # Add HELM if available
        helm = raw_data.get('HELM')
        if helm and not pd.isna(helm):
            assay_metadata['helm'] = str(helm)

        # Add molecule shape
        shape = raw_data.get('Molecule_Shape')
        if shape and not pd.isna(shape):
            assay_metadata['molecule_shape'] = str(shape)

        return assay_type, assay_metadata
