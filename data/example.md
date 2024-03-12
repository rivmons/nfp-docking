## Examples of file structure for data files

### Gibbs' Free Energy-valued files
* The Gibbs' Free Energy file will always be per protein since values vary based on protein, binding site, etc.

```
name: dock_acease.txt

1  -10.2	ZINC000278629005
2  -10.7	ZINC001168022560
3  -9.4		ZINC000846144591
...
```

### SMILES files
* If the dataset of molecules is different between molecules, you can put all smiles into a single file or alter train.py to choose the appropriate smiles file

```
name: smilesDS.smi

C[C@@H]([NH2+]CCC[N@@H+]1CCCO[C@H](C)C1)c1ccc2ccccc2c1O ZINC000278629005
C[C@H]1CC([NH2+]Cc2ccc(O)c(O)c2)C[C@H](C)[NH+]1Cc1ccccc1 ZINC001168022560
COc1cccc([C@H](C)[NH2+]CC[C@H](C)[N@@H+](C)Cc2ccccc2)c1O ZINC000846144591
...
```