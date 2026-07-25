# Probe60 Heat Mapping JNL Snippets

Verdict: `PASS_HEAT_MAPPING_JNL_PATTERNS_READY`

The snippets below are extracted from journal text only. They do not replace Abaqus CAE object inspection before generation.

## Inferred Patterns

| N | Heat sets | Scan template | Cool template | Load template | Can infer from JNL |
|---:|---:|---|---|---|---|
| 12 | 12 | `step_scan_00` | `step_cool_00` | `load_body_hflux_00` | `True` |
| 16 | 16 | `step_scan_00` | `step_cool_00` | `load_body_hflux_00` | `True` |
| 24 | 24 | `step_scan_00` | `step_cool_00` | `load_body_hflux_00` | `True` |
| 40 | 40 | `step_scan_00` | `step_cool_00` | `load_body_hflux_00` | `True` |

## Snippets

### N12 line 15 (mdb.models)

```text
0012: from sketch import *
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.06)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
```

### N12 line 16 (mdb.models)

```text
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.06)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.028, 0.003))
```

### N12 line 18 (mdb.models)

```text
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.06)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.028, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
```

### N12 line 20 (mdb.models)

```text
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.028, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
```

### N12 line 22 (mdb.models)

```text
0019:     point2=(0.028, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
```

### N12 line 23 (mdb.models)

```text
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.056, transform=
```

### N12 line 24 (mdb.models)

```text
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.056, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
```

### N12 line 25 (mdb.models)

```text
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.056, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
```

### N12 line 27 (mdb.models)

```text
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.056, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.014, 0.0015,
0030:     0.0)))
```

### N12 line 28 (mdb.models)

```text
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.056, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.014, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
```

### N12 line 31 (mdb.models)

```text
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.014, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
```

### N12 line 33 (mdb.models)

```text
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.014, -0.0015))
```

### N12 line 34 (mdb.models)

```text
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.014, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N12 line 35 (mdb.models)

```text
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.014, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
```

### N12 line 37 (mdb.models)

```text
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.014, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N12 line 39 (mdb.models)

```text
0036:     point1=(-0.014, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
```

### N12 line 40 (mdb.models)

```text
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N12 line 41 (mdb.models)

```text
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
```

### N12 line 43 (mdb.models)

```text
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0045:     point1=(-0.014, 0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N12 line 44 (mdb.models)

```text
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0045:     point1=(-0.014, 0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
```

### N12 line 46 (mdb.models)

```text
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0045:     point1=(-0.014, 0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N12 line 48 (mdb.models)

```text
0045:     point1=(-0.014, 0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
```

### N12 line 49 (mdb.models)

```text
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N12 line 50 (mdb.models)

```text
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
```

### N12 line 52 (mdb.models)

```text
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0054:     point1=(-0.014, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N12 line 53 (mdb.models)

```text
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0054:     point1=(-0.014, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
```

### N12 line 55 (mdb.models)

```text
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0054:     point1=(-0.014, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N12 line 57 (mdb.models)

```text
0054:     point1=(-0.014, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N12 line 58 (mdb.models)

```text
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0133141051381826,
```

### N12 line 59 (mdb.models)

```text
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0133141051381826,
0062:     0.0015), point2=(-0.0133141051381826, -0.00150000000651926))
```

### N12 line 60 (mdb.models)

```text
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0133141051381826,
0062:     0.0015), point2=(-0.0133141051381826, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
```

### N12 line 61 (mdb.models)

```text
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0133141051381826,
0062:     0.0015), point2=(-0.0133141051381826, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N12 line 63 (mdb.models)

```text
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0133141051381826,
0062:     0.0015), point2=(-0.0133141051381826, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
```

### N12 line 64 (mdb.models)

```text
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0133141051381826,
0062:     0.0015), point2=(-0.0133141051381826, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
```

### N12 line 65 (mdb.models)

```text
0062:     0.0015), point2=(-0.0133141051381826, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N12 line 67 (mdb.models)

```text
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
```

### N12 line 68 (mdb.models)

```text
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
```

### N12 line 69 (mdb.models)

```text
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
```

### N12 line 71 (mdb.models)

```text
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
```

### N12 line 72 (mdb.models)

```text
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
```

### N12 line 73 (mdb.models)

```text
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
```

### N12 line 75 (mdb.models)

```text
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
```

### N12 line 76 (mdb.models)

```text
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], textPoint=(
```

### N12 line 77 (mdb.models)

```text
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], textPoint=(
0080:     -0.0126581858247519, 0.00346947299689054), value=0.002)
```

### N12 line 78 (mdb.models)

```text
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], textPoint=(
0080:     -0.0126581858247519, 0.00346947299689054), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
```

### N12 line 79 (mdb.models)

```text
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], textPoint=(
0080:     -0.0126581858247519, 0.00346947299689054), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
```

### N12 line 81 (mdb.models)

```text
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], textPoint=(
0080:     -0.0126581858247519, 0.00346947299689054), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=13, number2=1, spacing1=0.002, spacing2=0.0056, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.014,
```

### N12 line 82 (mdb.models)

```text
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], textPoint=(
0080:     -0.0126581858247519, 0.00346947299689054), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=13, number2=1, spacing1=0.002, spacing2=0.0056, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.014,
0085:     0.000528883202001452), point2=(0.0139999999832362, 0.000528883202001452))
```

### N12 line 84 (mdb.models)

```text
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=13, number2=1, spacing1=0.002, spacing2=0.0056, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.014,
0085:     0.000528883202001452), point2=(0.0139999999832362, 0.000528883202001452))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
```

### N12 line 86 (mdb.models)

```text
0083:     ), number1=13, number2=1, spacing1=0.002, spacing2=0.0056, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.014,
0085:     0.000528883202001452), point2=(0.0139999999832362, 0.000528883202001452))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
```

### N12 line 88 (mdb.models)

```text
0085:     0.000528883202001452), point2=(0.0139999999832362, 0.000528883202001452))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
```

### N12 line 89 (mdb.models)

```text
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
```

### N12 line 91 (mdb.models)

```text
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
```

### N12 line 92 (mdb.models)

```text
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[30], entity2=
```

### N12 line 93 (mdb.models)

```text
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[30], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
```

### N12 line 95 (mdb.models)

```text
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[22])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[30], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
```

### N12 line 96 (mdb.models)

```text
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[30], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[31], entity2=
```

### N12 line 97 (mdb.models)

```text
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[30], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[31], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
```

### N12 line 99 (mdb.models)

```text
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[31], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[22], entity2=
```

### N12 line 100 (mdb.models)

```text
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[31], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[22], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
```

### N12 line 101 (mdb.models)

```text
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[31], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[22], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0142345088869333, -0.000941411579027772), value=0.002)
```

### N12 line 102 (mdb.models)

```text
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[31], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[22], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0142345088869333, -0.000941411579027772), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
```

### N12 line 103 (mdb.models)

```text
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[22], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0142345088869333, -0.000941411579027772), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 105 (mdb.models)

```text
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[22], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0142345088869333, -0.000941411579027772), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
```

### N12 line 106 (mdb.models)

```text
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0142345088869333, -0.000941411579027772), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].Material(name='SS316L For AM')
```

### N12 line 107 (mdb.models)

```text
0104:     -0.0142345088869333, -0.000941411579027772), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].Material(name='SS316L For AM')
0110: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
```

### N12 line 108 (mdb.models)

```text
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].Material(name='SS316L For AM')
0110: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0111:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
```

### N12 line 109 (mdb.models)

```text
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].Material(name='SS316L For AM')
0110: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0111:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0112:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
```

### N12 line 110 (mdb.models)

```text
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].Material(name='SS316L For AM')
0110: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0111:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0112:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0113:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
```

### N12 line 115 (mdb.models)

```text
0112:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0113:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
0114:     type=ISOTROPIC)
0115: mdb.models['Model-1'].materials['SS316L For AM'].Density(dependencies=0,
0116:     distributionType=UNIFORM, fieldName='', table=((7980.0, 20.0), (7950.0,
0117:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0118:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
```

### N12 line 120 (mdb.models)

```text
0117:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0118:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
0119:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0120: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0121:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0122: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0123:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
```

### N12 line 122 (mdb.models)

```text
0119:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0120: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0121:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0122: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0123:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
0124:     0.3, 20.0), (202000000000.0, 0.3, 100.0), (194000000000.0, 0.3, 200.0), (
0125:     178000000000.0, 0.3, 400.0), (162000000000.0, 0.3, 600.0), (142000000000.0,
```

### N12 line 130 (mdb.models)

```text
0127:     15000000000.0, 0.3, 1375.0), (100000000.0, 0.3, 1400.0), (10000000.0, 0.3,
0128:     1723.0), (1000000.0, 0.3, 3000.0)), temperatureDependency=ON, type=
0129:     ISOTROPIC)
0130: mdb.models['Model-1'].materials['SS316L For AM'].Expansion(dependencies=0,
0131:     table=((1.48e-05, 20.0), (1.6e-05, 100.0), (1.68e-05, 200.0), (1.78e-05,
0132:     400.0), (1.87e-05, 600.0), (1.96e-05, 800.0), (2.02e-05, 1000.0), (
0133:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
```

### N12 line 136 (mdb.models)

```text
0133:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
0134:     1723.0), (2.2e-05, 3000.0)), temperatureDependency=ON, type=ISOTROPIC,
0135:     userSubroutine=OFF, zero=0.0)
0136: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0137:     1375.0, 1400.0), ))
0138: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0139:     '')
```

### N12 line 138 (mdb.models)

```text
0135:     userSubroutine=OFF, zero=0.0)
0136: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0137:     1375.0, 1400.0), ))
0138: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0139:     '')
0140: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0141:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
```

### N12 line 140 (mdb.models)

```text
0137:     1375.0, 1400.0), ))
0138: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0139:     '')
0140: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0141:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
0142:     numBackstresses=1, rate=OFF, scaleStress=None, staticRecovery=OFF,
0143:     strainRangeDependency=OFF, table=((580000000.0, 0.0, 20.0), (530000000.0,
```

### N12 line 149 (mdb.models)

```text
0146:     1000.0), (30000000.0, 0.0, 1200.0), (2000000.0, 0.0, 1375.0), (10000.0,
0147:     0.0, 1400.0), (5000.0, 0.0, 1723.0), (1000.0, 0.0, 3000.0)),
0148:     temperatureDependency=ON)
0149: mdb.models['Model-1'].materials['SS316L For AM'].SpecificHeat(dependencies=0,
0150:     law=CONSTANTVOLUME, table=((450.0, 20.0), (480.0, 100.0), (505.0, 200.0), (
0151:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0152:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
```

### N12 line 154 (mdb.models)

```text
0151:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0152:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
0153:     3000.0)), temperatureDependency=ON)
0154: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0155:     'property_section_all', thickness=None)
0156: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0157:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 156 (mdb.models)

```text
0153:     3000.0)), temperatureDependency=ON)
0154: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0155:     'property_section_all', thickness=None)
0156: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0157:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0158:     '[#fffffff ]', ), ), name='section_all')
0159: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
```

### N12 line 157 (mdb.models)

```text
0154: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0155:     'property_section_all', thickness=None)
0156: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0157:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0158:     '[#fffffff ]', ), ), name='section_all')
0159: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0160:     offsetField='', offsetType=MIDDLE_SURFACE, region=
```

### N12 line 159 (mdb.models)

```text
0156: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0157:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0158:     '[#fffffff ]', ), ), name='section_all')
0159: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0160:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0161:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0162:     'property_section_all', thicknessAssignment=FROM_SECTION)
```

### N12 line 160 (region=)

```text
0157:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0158:     '[#fffffff ]', ), ), name='section_all')
0159: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0160:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0161:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0162:     'property_section_all', thicknessAssignment=FROM_SECTION)
0163: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
```

### N12 line 161 (mdb.models)

```text
0158:     '[#fffffff ]', ), ), name='section_all')
0159: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0160:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0161:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0162:     'property_section_all', thicknessAssignment=FROM_SECTION)
0163: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0164: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
```

### N12 line 163 (mdb.models)

```text
0160:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0161:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0162:     'property_section_all', thicknessAssignment=FROM_SECTION)
0163: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0164: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0165:     part=mdb.models['Model-1'].parts['part_plate'])
0166: mdb.models['Model-1'].setValues(absoluteZero=-173, stefanBoltzmann=5.67e-08)
```

### N12 line 164 (mdb.models)

```text
0161:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0162:     'property_section_all', thicknessAssignment=FROM_SECTION)
0163: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0164: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0165:     part=mdb.models['Model-1'].parts['part_plate'])
0166: mdb.models['Model-1'].setValues(absoluteZero=-173, stefanBoltzmann=5.67e-08)
0167: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 165 (mdb.models)

```text
0162:     'property_section_all', thicknessAssignment=FROM_SECTION)
0163: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0164: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0165:     part=mdb.models['Model-1'].parts['part_plate'])
0166: mdb.models['Model-1'].setValues(absoluteZero=-173, stefanBoltzmann=5.67e-08)
0167: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0168:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 166 (mdb.models)

```text
0163: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0164: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0165:     part=mdb.models['Model-1'].parts['part_plate'])
0166: mdb.models['Model-1'].setValues(absoluteZero=-173, stefanBoltzmann=5.67e-08)
0167: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0168:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0169:     '[#4000000 ]', ), ), name='set_body_heat_00')
```

### N12 line 167 (mdb.models)

```text
0164: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0165:     part=mdb.models['Model-1'].parts['part_plate'])
0166: mdb.models['Model-1'].setValues(absoluteZero=-173, stefanBoltzmann=5.67e-08)
0167: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0168:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0169:     '[#4000000 ]', ), ), name='set_body_heat_00')
0170: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 168 (mdb.models)

```text
0165:     part=mdb.models['Model-1'].parts['part_plate'])
0166: mdb.models['Model-1'].setValues(absoluteZero=-173, stefanBoltzmann=5.67e-08)
0167: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0168:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0169:     '[#4000000 ]', ), ), name='set_body_heat_00')
0170: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0171:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 169 (set_body_heat_)

```text
0166: mdb.models['Model-1'].setValues(absoluteZero=-173, stefanBoltzmann=5.67e-08)
0167: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0168:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0169:     '[#4000000 ]', ), ), name='set_body_heat_00')
0170: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0171:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0172:     '[#800000 ]', ), ), name='set_body_heat_01')
```

### N12 line 170 (mdb.models)

```text
0167: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0168:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0169:     '[#4000000 ]', ), ), name='set_body_heat_00')
0170: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0171:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0172:     '[#800000 ]', ), ), name='set_body_heat_01')
0173: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 171 (mdb.models)

```text
0168:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0169:     '[#4000000 ]', ), ), name='set_body_heat_00')
0170: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0171:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0172:     '[#800000 ]', ), ), name='set_body_heat_01')
0173: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0174:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 172 (set_body_heat_)

```text
0169:     '[#4000000 ]', ), ), name='set_body_heat_00')
0170: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0171:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0172:     '[#800000 ]', ), ), name='set_body_heat_01')
0173: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0174:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0175:     '[#200000 ]', ), ), name='set_body_heat_02')
```

### N12 line 173 (mdb.models)

```text
0170: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0171:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0172:     '[#800000 ]', ), ), name='set_body_heat_01')
0173: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0174:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0175:     '[#200000 ]', ), ), name='set_body_heat_02')
0176: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 174 (mdb.models)

```text
0171:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0172:     '[#800000 ]', ), ), name='set_body_heat_01')
0173: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0174:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0175:     '[#200000 ]', ), ), name='set_body_heat_02')
0176: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0177:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 175 (set_body_heat_)

```text
0172:     '[#800000 ]', ), ), name='set_body_heat_01')
0173: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0174:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0175:     '[#200000 ]', ), ), name='set_body_heat_02')
0176: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0177:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0178:     '[#80000 ]', ), ), name='set_body_heat_03')
```

### N12 line 176 (mdb.models)

```text
0173: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0174:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0175:     '[#200000 ]', ), ), name='set_body_heat_02')
0176: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0177:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0178:     '[#80000 ]', ), ), name='set_body_heat_03')
0179: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 177 (mdb.models)

```text
0174:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0175:     '[#200000 ]', ), ), name='set_body_heat_02')
0176: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0177:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0178:     '[#80000 ]', ), ), name='set_body_heat_03')
0179: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0180:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 178 (set_body_heat_)

```text
0175:     '[#200000 ]', ), ), name='set_body_heat_02')
0176: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0177:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0178:     '[#80000 ]', ), ), name='set_body_heat_03')
0179: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0180:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0181:     '[#20000 ]', ), ), name='set_body_heat_04')
```

### N12 line 179 (mdb.models)

```text
0176: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0177:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0178:     '[#80000 ]', ), ), name='set_body_heat_03')
0179: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0180:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0181:     '[#20000 ]', ), ), name='set_body_heat_04')
0182: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 180 (mdb.models)

```text
0177:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0178:     '[#80000 ]', ), ), name='set_body_heat_03')
0179: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0180:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0181:     '[#20000 ]', ), ), name='set_body_heat_04')
0182: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0183:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 181 (set_body_heat_)

```text
0178:     '[#80000 ]', ), ), name='set_body_heat_03')
0179: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0180:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0181:     '[#20000 ]', ), ), name='set_body_heat_04')
0182: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0183:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0184:     '[#8000 ]', ), ), name='set_body_heat_05')
```

### N12 line 182 (mdb.models)

```text
0179: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0180:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0181:     '[#20000 ]', ), ), name='set_body_heat_04')
0182: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0183:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0184:     '[#8000 ]', ), ), name='set_body_heat_05')
0185: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 183 (mdb.models)

```text
0180:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0181:     '[#20000 ]', ), ), name='set_body_heat_04')
0182: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0183:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0184:     '[#8000 ]', ), ), name='set_body_heat_05')
0185: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0186:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 184 (set_body_heat_)

```text
0181:     '[#20000 ]', ), ), name='set_body_heat_04')
0182: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0183:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0184:     '[#8000 ]', ), ), name='set_body_heat_05')
0185: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0186:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0187:     '[#2000 ]', ), ), name='set_body_heat_06')
```

### N12 line 185 (mdb.models)

```text
0182: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0183:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0184:     '[#8000 ]', ), ), name='set_body_heat_05')
0185: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0186:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0187:     '[#2000 ]', ), ), name='set_body_heat_06')
0188: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 186 (mdb.models)

```text
0183:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0184:     '[#8000 ]', ), ), name='set_body_heat_05')
0185: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0186:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0187:     '[#2000 ]', ), ), name='set_body_heat_06')
0188: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0189:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 187 (set_body_heat_)

```text
0184:     '[#8000 ]', ), ), name='set_body_heat_05')
0185: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0186:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0187:     '[#2000 ]', ), ), name='set_body_heat_06')
0188: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0189:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0190:     '[#800 ]', ), ), name='set_body_heat_07')
```

### N12 line 188 (mdb.models)

```text
0185: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0186:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0187:     '[#2000 ]', ), ), name='set_body_heat_06')
0188: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0189:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0190:     '[#800 ]', ), ), name='set_body_heat_07')
0191: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 189 (mdb.models)

```text
0186:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0187:     '[#2000 ]', ), ), name='set_body_heat_06')
0188: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0189:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0190:     '[#800 ]', ), ), name='set_body_heat_07')
0191: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0192:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 190 (set_body_heat_)

```text
0187:     '[#2000 ]', ), ), name='set_body_heat_06')
0188: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0189:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0190:     '[#800 ]', ), ), name='set_body_heat_07')
0191: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0192:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0193:     '[#200 ]', ), ), name='set_body_heat_08')
```

### N12 line 191 (mdb.models)

```text
0188: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0189:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0190:     '[#800 ]', ), ), name='set_body_heat_07')
0191: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0192:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0193:     '[#200 ]', ), ), name='set_body_heat_08')
0194: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 192 (mdb.models)

```text
0189:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0190:     '[#800 ]', ), ), name='set_body_heat_07')
0191: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0192:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0193:     '[#200 ]', ), ), name='set_body_heat_08')
0194: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0195:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 193 (set_body_heat_)

```text
0190:     '[#800 ]', ), ), name='set_body_heat_07')
0191: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0192:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0193:     '[#200 ]', ), ), name='set_body_heat_08')
0194: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0195:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0196:     '[#80 ]', ), ), name='set_body_heat_09')
```

### N12 line 194 (mdb.models)

```text
0191: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0192:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0193:     '[#200 ]', ), ), name='set_body_heat_08')
0194: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0195:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0196:     '[#80 ]', ), ), name='set_body_heat_09')
0197: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 195 (mdb.models)

```text
0192:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0193:     '[#200 ]', ), ), name='set_body_heat_08')
0194: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0195:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0196:     '[#80 ]', ), ), name='set_body_heat_09')
0197: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0198:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 196 (set_body_heat_)

```text
0193:     '[#200 ]', ), ), name='set_body_heat_08')
0194: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0195:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0196:     '[#80 ]', ), ), name='set_body_heat_09')
0197: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0198:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0199:     '[#20 ]', ), ), name='set_body_heat_10')
```

### N12 line 197 (mdb.models)

```text
0194: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0195:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0196:     '[#80 ]', ), ), name='set_body_heat_09')
0197: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0198:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0199:     '[#20 ]', ), ), name='set_body_heat_10')
0200: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N12 line 198 (mdb.models)

```text
0195:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0196:     '[#80 ]', ), ), name='set_body_heat_09')
0197: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0198:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0199:     '[#20 ]', ), ), name='set_body_heat_10')
0200: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0201:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 199 (set_body_heat_)

```text
0196:     '[#80 ]', ), ), name='set_body_heat_09')
0197: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0198:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0199:     '[#20 ]', ), ), name='set_body_heat_10')
0200: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0201:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0202:     '[#8 ]', ), ), name='set_body_heat_11')
```

### N12 line 200 (mdb.models)

```text
0197: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0198:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0199:     '[#20 ]', ), ), name='set_body_heat_10')
0200: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0201:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0202:     '[#8 ]', ), ), name='set_body_heat_11')
0203: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
```

### N12 line 201 (mdb.models)

```text
0198:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0199:     '[#20 ]', ), ), name='set_body_heat_10')
0200: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0201:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0202:     '[#8 ]', ), ), name='set_body_heat_11')
0203: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0204:     side1Edges=
```

### N12 line 202 (set_body_heat_)

```text
0199:     '[#20 ]', ), ), name='set_body_heat_10')
0200: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0201:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0202:     '[#8 ]', ), ), name='set_body_heat_11')
0203: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0204:     side1Edges=
0205:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
```

### N12 line 203 (mdb.models)

```text
0200: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0201:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0202:     '[#8 ]', ), ), name='set_body_heat_11')
0203: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0204:     side1Edges=
0205:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0206:     '[#94a54966 #a5294a52 #ec ]', ), ))
```

### N12 line 205 (mdb.models)

```text
0202:     '[#8 ]', ), ), name='set_body_heat_11')
0203: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0204:     side1Edges=
0205:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0206:     '[#94a54966 #a5294a52 #ec ]', ), ))
0207: mdb.models['Model-1'].rootAssembly.regenerate()
0208: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.2,
```

### N12 line 207 (mdb.models)

```text
0204:     side1Edges=
0205:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0206:     '[#94a54966 #a5294a52 #ec ]', ), ))
0207: mdb.models['Model-1'].rootAssembly.regenerate()
0208: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.2,
0209:     maxInc=0.2, maxNumInc=999999, minInc=2e-06, name='step_scan_00', nlgeom=ON,
0210:     previous='Initial', timePeriod=0.2)
```

### N12 line 208 (mdb.models;CoupledTempDisplacementStep)

```text
0205:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0206:     '[#94a54966 #a5294a52 #ec ]', ), ))
0207: mdb.models['Model-1'].rootAssembly.regenerate()
0208: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.2,
0209:     maxInc=0.2, maxNumInc=999999, minInc=2e-06, name='step_scan_00', nlgeom=ON,
0210:     previous='Initial', timePeriod=0.2)
0211: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
```

### N12 line 209 (step_scan_)

```text
0206:     '[#94a54966 #a5294a52 #ec ]', ), ))
0207: mdb.models['Model-1'].rootAssembly.regenerate()
0208: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.2,
0209:     maxInc=0.2, maxNumInc=999999, minInc=2e-06, name='step_scan_00', nlgeom=ON,
0210:     previous='Initial', timePeriod=0.2)
0211: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0212:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
```

### N12 line 211 (mdb.models;CoupledTempDisplacementStep)

```text
0208: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.2,
0209:     maxInc=0.2, maxNumInc=999999, minInc=2e-06, name='step_scan_00', nlgeom=ON,
0210:     previous='Initial', timePeriod=0.2)
0211: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0212:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0213:     previous='step_scan_00', timePeriod=3.4)
0214: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
```

### N12 line 212 (step_cool_)

```text
0209:     maxInc=0.2, maxNumInc=999999, minInc=2e-06, name='step_scan_00', nlgeom=ON,
0210:     previous='Initial', timePeriod=0.2)
0211: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0212:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0213:     previous='step_scan_00', timePeriod=3.4)
0214: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0215:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
```

### N12 line 213 (step_scan_)

```text
0210:     previous='Initial', timePeriod=0.2)
0211: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0212:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0213:     previous='step_scan_00', timePeriod=3.4)
0214: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0215:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0216: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
```

### N12 line 214 (mdb.models)

```text
0211: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0212:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0213:     previous='step_scan_00', timePeriod=3.4)
0214: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0215:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0216: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0217:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
```

### N12 line 216 (mdb.models)

```text
0213:     previous='step_scan_00', timePeriod=3.4)
0214: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0215:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0216: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0217:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0218: mdb.models['Model-1'].FilmCondition(createStepName='step_scan_00', definition=
0219:     EMBEDDED_COEFF, filmCoeff=46.3, filmCoeffAmplitude='', name=
```

### N12 line 218 (step_scan_;mdb.models;createStepName)

```text
0215:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0216: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0217:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0218: mdb.models['Model-1'].FilmCondition(createStepName='step_scan_00', definition=
0219:     EMBEDDED_COEFF, filmCoeff=46.3, filmCoeffAmplitude='', name=
0220:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0221:     sinkFieldName='', sinkTemperature=20.0, surface=
```

### N12 line 222 (mdb.models)

```text
0219:     EMBEDDED_COEFF, filmCoeff=46.3, filmCoeffAmplitude='', name=
0220:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0221:     sinkFieldName='', sinkTemperature=20.0, surface=
0222:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0223: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0224:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0225:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
```

### N12 line 223 (mdb.models)

```text
0220:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0221:     sinkFieldName='', sinkTemperature=20.0, surface=
0222:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0223: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0224:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0225:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0226:     radiationType=AMBIENT, surface=
```

### N12 line 224 (step_scan_;createStepName)

```text
0221:     sinkFieldName='', sinkTemperature=20.0, surface=
0222:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0223: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0224:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0225:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0226:     radiationType=AMBIENT, surface=
0227:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
```

### N12 line 227 (mdb.models)

```text
0224:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0225:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0226:     radiationType=AMBIENT, surface=
0227:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0228: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0229:     80000000000.0, name='load_body_hflux_00', region=
0230:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
```

### N12 line 228 (step_scan_;BodyHeatFlux;mdb.models;createStepName)

```text
0225:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0226:     radiationType=AMBIENT, surface=
0227:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0228: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0229:     80000000000.0, name='load_body_hflux_00', region=
0230:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0231: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N12 line 229 (load_body_hflux_;region=)

```text
0226:     radiationType=AMBIENT, surface=
0227:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0228: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0229:     80000000000.0, name='load_body_hflux_00', region=
0230:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0231: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0232:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
```

### N12 line 230 (set_body_heat_;mdb.models)

```text
0227:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0228: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0229:     80000000000.0, name='load_body_hflux_00', region=
0230:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0231: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0232:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0233:     region=Region(
```

### N12 line 231 (mdb.models;createStepName)

```text
0228: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0229:     80000000000.0, name='load_body_hflux_00', region=
0230:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0231: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0232:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0233:     region=Region(
0234:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N12 line 233 (region=;Region)

```text
0230:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0231: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0232:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0233:     region=Region(
0234:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0235:     mask=('[#0 #400 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0236: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N12 line 234 (mdb.models)

```text
0231: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0232:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0233:     region=Region(
0234:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0235:     mask=('[#0 #400 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0236: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0237:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
```

### N12 line 236 (mdb.models;createStepName)

```text
0233:     region=Region(
0234:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0235:     mask=('[#0 #400 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0236: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0237:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0238:     region=Region(
0239:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N12 line 238 (region=;Region)

```text
0235:     mask=('[#0 #400 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0236: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0237:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0238:     region=Region(
0239:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0240:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0241: mdb.models['Model-1'].Temperature(createStepName='Initial',
```

### N12 line 239 (mdb.models)

```text
0236: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0237:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0238:     region=Region(
0239:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0240:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0241: mdb.models['Model-1'].Temperature(createStepName='Initial',
0242:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
```

### N12 line 241 (mdb.models;createStepName)

```text
0238:     region=Region(
0239:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0240:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0241: mdb.models['Model-1'].Temperature(createStepName='Initial',
0242:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0243:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0244:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
```

### N12 line 243 (region=)

```text
0240:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0241: mdb.models['Model-1'].Temperature(createStepName='Initial',
0242:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0243:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0244:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0245: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0246:     minSizeFactor=0.1, size=0.0005)
```

### N12 line 244 (mdb.models)

```text
0241: mdb.models['Model-1'].Temperature(createStepName='Initial',
0242:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0243:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0244:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0245: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0246:     minSizeFactor=0.1, size=0.0005)
0247: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
```

### N12 line 245 (mdb.models)

```text
0242:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0243:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0244:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0245: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0246:     minSizeFactor=0.1, size=0.0005)
0247: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0248:     regions=
```

### N12 line 247 (mdb.models)

```text
0244:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0245: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0246:     minSizeFactor=0.1, size=0.0005)
0247: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0248:     regions=
0249:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0250:     '[#fffffff ]', ), ), technique=STRUCTURED)
```

### N12 line 249 (mdb.models)

```text
0246:     minSizeFactor=0.1, size=0.0005)
0247: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0248:     regions=
0249:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0250:     '[#fffffff ]', ), ), technique=STRUCTURED)
0251: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0252:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
```

### N12 line 251 (mdb.models)

```text
0248:     regions=
0249:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0250:     '[#fffffff ]', ), ), technique=STRUCTURED)
0251: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0252:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0253:     elemLibrary=STANDARD)), regions=(
0254:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N12 line 254 (mdb.models)

```text
0251: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0252:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0253:     elemLibrary=STANDARD)), regions=(
0254:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0255:     '[#fffffff ]', ), ), ))
0256: mdb.models['Model-1'].parts['part_plate'].generateMesh()
0257: mdb.models['Model-1'].rootAssembly.regenerate()
```

### N12 line 256 (mdb.models)

```text
0253:     elemLibrary=STANDARD)), regions=(
0254:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0255:     '[#fffffff ]', ), ), ))
0256: mdb.models['Model-1'].parts['part_plate'].generateMesh()
0257: mdb.models['Model-1'].rootAssembly.regenerate()
0258: # Save by wuxia on 2026_06_10-21.25.34; build 2024 2023_09_21-20.55.25 RELr426 190762
0259: from part import *
```

### N12 line 257 (mdb.models)

```text
0254:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0255:     '[#fffffff ]', ), ), ))
0256: mdb.models['Model-1'].parts['part_plate'].generateMesh()
0257: mdb.models['Model-1'].rootAssembly.regenerate()
0258: # Save by wuxia on 2026_06_10-21.25.34; build 2024 2023_09_21-20.55.25 RELr426 190762
0259: from part import *
0260: from material import *
```

### N12 line 272 (step_cool_;load_body_hflux_;mdb.models;loads[;deactivate)

```text
0269: from sketch import *
0270: from visualization import *
0271: from connectorBehavior import *
0272: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0273: # Save by wuxia on 2026_06_10-23.00.34; build 2024 2023_09_21-20.55.25 RELr426 190762
```

### N16 line 15 (mdb.models)

```text
0012: from sketch import *
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.08)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
```

### N16 line 16 (mdb.models)

```text
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.08)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.036, 0.003))
```

### N16 line 18 (mdb.models)

```text
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.08)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.036, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
```

### N16 line 20 (mdb.models)

```text
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.036, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
```

### N16 line 22 (mdb.models)

```text
0019:     point2=(0.036, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
```

### N16 line 23 (mdb.models)

```text
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.072, transform=
```

### N16 line 24 (mdb.models)

```text
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.072, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
```

### N16 line 25 (mdb.models)

```text
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.072, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
```

### N16 line 27 (mdb.models)

```text
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.072, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.018, 0.0015,
0030:     0.0)))
```

### N16 line 28 (mdb.models)

```text
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.001, name='__profile__',
0026:     sheetSize=0.072, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.018, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
```

### N16 line 31 (mdb.models)

```text
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.018, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
```

### N16 line 33 (mdb.models)

```text
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.018, -0.0015))
```

### N16 line 34 (mdb.models)

```text
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.018, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N16 line 35 (mdb.models)

```text
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.018, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
```

### N16 line 37 (mdb.models)

```text
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.018, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N16 line 39 (mdb.models)

```text
0036:     point1=(-0.018, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N16 line 40 (mdb.models)

```text
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
```

### N16 line 41 (mdb.models)

```text
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.018, -0.0015))
```

### N16 line 42 (mdb.models)

```text
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.018, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N16 line 43 (mdb.models)

```text
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.018, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
```

### N16 line 45 (mdb.models)

```text
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.018, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N16 line 47 (mdb.models)

```text
0044:     point1=(-0.018, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
```

### N16 line 48 (mdb.models)

```text
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N16 line 49 (mdb.models)

```text
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
```

### N16 line 51 (mdb.models)

```text
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0053:     point1=(-0.018, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N16 line 52 (mdb.models)

```text
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0053:     point1=(-0.018, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
```

### N16 line 54 (mdb.models)

```text
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0053:     point1=(-0.018, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N16 line 56 (mdb.models)

```text
0053:     point1=(-0.018, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
```

### N16 line 57 (mdb.models)

```text
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N16 line 58 (mdb.models)

```text
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0166556656658649,
```

### N16 line 60 (mdb.models)

```text
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0166556656658649,
0062:     0.0015), point2=(-0.0166556656658649, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
```

### N16 line 61 (mdb.models)

```text
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0166556656658649,
0062:     0.0015), point2=(-0.0166556656658649, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N16 line 63 (mdb.models)

```text
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0166556656658649,
0062:     0.0015), point2=(-0.0166556656658649, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
```

### N16 line 64 (mdb.models)

```text
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0166556656658649,
0062:     0.0015), point2=(-0.0166556656658649, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
```

### N16 line 65 (mdb.models)

```text
0062:     0.0015), point2=(-0.0166556656658649, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N16 line 67 (mdb.models)

```text
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
```

### N16 line 68 (mdb.models)

```text
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
```

### N16 line 69 (mdb.models)

```text
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
```

### N16 line 71 (mdb.models)

```text
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
```

### N16 line 72 (mdb.models)

```text
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
```

### N16 line 73 (mdb.models)

```text
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
```

### N16 line 75 (mdb.models)

```text
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
```

### N16 line 76 (mdb.models)

```text
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
```

### N16 line 77 (mdb.models)

```text
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0166556656658649, 0.00416155277192593), value=0.002)
```

### N16 line 78 (mdb.models)

```text
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0166556656658649, 0.00416155277192593), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
```

### N16 line 79 (mdb.models)

```text
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0166556656658649, 0.00416155277192593), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
```

### N16 line 81 (mdb.models)

```text
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0166556656658649, 0.00416155277192593), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=17, number2=1, spacing1=0.002, spacing2=0.0072, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.018,
```

### N16 line 82 (mdb.models)

```text
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0166556656658649, 0.00416155277192593), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=17, number2=1, spacing1=0.002, spacing2=0.0072, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.018,
0085:     0.000666392048820853), point2=(0.0179999999618158, 0.000666392048820853))
```

### N16 line 84 (mdb.models)

```text
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=17, number2=1, spacing1=0.002, spacing2=0.0072, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.018,
0085:     0.000666392048820853), point2=(0.0179999999618158, 0.000666392048820853))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
```

### N16 line 86 (mdb.models)

```text
0083:     ), number1=17, number2=1, spacing1=0.002, spacing2=0.0072, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.018,
0085:     0.000666392048820853), point2=(0.0179999999618158, 0.000666392048820853))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
```

### N16 line 88 (mdb.models)

```text
0085:     0.000666392048820853), point2=(0.0179999999618158, 0.000666392048820853))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
```

### N16 line 89 (mdb.models)

```text
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
```

### N16 line 91 (mdb.models)

```text
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
```

### N16 line 92 (mdb.models)

```text
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[38], entity2=
```

### N16 line 93 (mdb.models)

```text
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[38], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
```

### N16 line 95 (mdb.models)

```text
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[26])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[38], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
```

### N16 line 96 (mdb.models)

```text
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[38], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[39], entity2=
```

### N16 line 97 (mdb.models)

```text
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[38], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[39], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
```

### N16 line 99 (mdb.models)

```text
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[39], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[26], entity2=
```

### N16 line 100 (mdb.models)

```text
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[39], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[26], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
```

### N16 line 101 (mdb.models)

```text
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[39], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[26], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0183287140280008, -0.000530393781140447), value=0.002)
```

### N16 line 102 (mdb.models)

```text
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[39], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[26], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0183287140280008, -0.000530393781140447), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
```

### N16 line 103 (mdb.models)

```text
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[26], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0183287140280008, -0.000530393781140447), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 105 (mdb.models)

```text
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[26], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0183287140280008, -0.000530393781140447), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
```

### N16 line 106 (mdb.models)

```text
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0183287140280008, -0.000530393781140447), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 107 (mdb.models)

```text
0104:     -0.0183287140280008, -0.000530393781140447), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 108 (mdb.models)

```text
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #f ]', ), ), name='section_all')
```

### N16 line 109 (mdb.models)

```text
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #f ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 110 (mdb.models)

```text
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #f ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 112 (mdb.models)

```text
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #f ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #4 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 113 (mdb.models)

```text
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #f ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #4 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 114 (set_body_heat_)

```text
0111:     '[#ffffffff #f ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #4 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#80000000 ]', ), ), name='set_body_heat_01')
```

### N16 line 115 (mdb.models)

```text
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #4 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#80000000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 116 (mdb.models)

```text
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #4 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#80000000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 117 (set_body_heat_)

```text
0114:     '[#0 #4 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#80000000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#20000000 ]', ), ), name='set_body_heat_02')
```

### N16 line 118 (mdb.models)

```text
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#80000000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#20000000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 119 (mdb.models)

```text
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#80000000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#20000000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 120 (set_body_heat_)

```text
0117:     '[#80000000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#20000000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#8000000 ]', ), ), name='set_body_heat_03')
```

### N16 line 121 (mdb.models)

```text
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#20000000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#8000000 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 122 (mdb.models)

```text
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#20000000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#8000000 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 123 (set_body_heat_)

```text
0120:     '[#20000000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#8000000 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#2000000 ]', ), ), name='set_body_heat_04')
```

### N16 line 124 (mdb.models)

```text
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#8000000 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#2000000 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 125 (mdb.models)

```text
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#8000000 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#2000000 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 126 (set_body_heat_)

```text
0123:     '[#8000000 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#2000000 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#800000 ]', ), ), name='set_body_heat_05')
```

### N16 line 127 (mdb.models)

```text
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#2000000 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#800000 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 128 (mdb.models)

```text
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#2000000 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#800000 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 129 (set_body_heat_)

```text
0126:     '[#2000000 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#800000 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#200000 ]', ), ), name='set_body_heat_06')
```

### N16 line 130 (mdb.models)

```text
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#800000 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#200000 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 131 (mdb.models)

```text
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#800000 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#200000 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 132 (set_body_heat_)

```text
0129:     '[#800000 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#200000 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#80000 ]', ), ), name='set_body_heat_07')
```

### N16 line 133 (mdb.models)

```text
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#200000 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#80000 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 134 (mdb.models)

```text
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#200000 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#80000 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 135 (set_body_heat_)

```text
0132:     '[#200000 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#80000 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#20000 ]', ), ), name='set_body_heat_08')
```

### N16 line 136 (mdb.models)

```text
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#80000 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#20000 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 137 (mdb.models)

```text
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#80000 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#20000 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 138 (set_body_heat_)

```text
0135:     '[#80000 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#20000 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#8000 ]', ), ), name='set_body_heat_09')
```

### N16 line 139 (mdb.models)

```text
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#20000 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#8000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 140 (mdb.models)

```text
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#20000 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#8000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 141 (set_body_heat_)

```text
0138:     '[#20000 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#8000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#2000 ]', ), ), name='set_body_heat_10')
```

### N16 line 142 (mdb.models)

```text
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#8000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#2000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 143 (mdb.models)

```text
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#8000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#2000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 144 (set_body_heat_)

```text
0141:     '[#8000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#2000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#800 ]', ), ), name='set_body_heat_11')
```

### N16 line 145 (mdb.models)

```text
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#2000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#800 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 146 (mdb.models)

```text
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#2000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#800 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 147 (set_body_heat_)

```text
0144:     '[#2000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#800 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#200 ]', ), ), name='set_body_heat_12')
```

### N16 line 148 (mdb.models)

```text
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#800 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#200 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 149 (mdb.models)

```text
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#800 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#200 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 150 (set_body_heat_)

```text
0147:     '[#800 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#200 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#80 ]', ), ), name='set_body_heat_13')
```

### N16 line 151 (mdb.models)

```text
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#200 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#80 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 152 (mdb.models)

```text
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#200 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#80 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 153 (set_body_heat_)

```text
0150:     '[#200 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#80 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#20 ]', ), ), name='set_body_heat_14')
```

### N16 line 154 (mdb.models)

```text
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#80 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#20 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N16 line 155 (mdb.models)

```text
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#80 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#20 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 156 (set_body_heat_)

```text
0153:     '[#80 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#20 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#8 ]', ), ), name='set_body_heat_15')
```

### N16 line 157 (mdb.models)

```text
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#20 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#8 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
```

### N16 line 158 (mdb.models)

```text
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#20 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#8 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0161:     side1Edges=
```

### N16 line 159 (set_body_heat_)

```text
0156:     '[#20 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#8 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0161:     side1Edges=
0162:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
```

### N16 line 160 (mdb.models)

```text
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#8 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0161:     side1Edges=
0162:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0163:     '[#94a54966 #a5294a52 #eca5294 ]', ), ))
```

### N16 line 162 (mdb.models)

```text
0159:     '[#8 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0161:     side1Edges=
0162:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0163:     '[#94a54966 #a5294a52 #eca5294 ]', ), ))
0164: mdb.models['Model-1'].Material(name='SS316L For AM')
0165: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
```

### N16 line 164 (mdb.models)

```text
0161:     side1Edges=
0162:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0163:     '[#94a54966 #a5294a52 #eca5294 ]', ), ))
0164: mdb.models['Model-1'].Material(name='SS316L For AM')
0165: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0166:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0167:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
```

### N16 line 165 (mdb.models)

```text
0162:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0163:     '[#94a54966 #a5294a52 #eca5294 ]', ), ))
0164: mdb.models['Model-1'].Material(name='SS316L For AM')
0165: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0166:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0167:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0168:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
```

### N16 line 170 (mdb.models)

```text
0167:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0168:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
0169:     type=ISOTROPIC)
0170: mdb.models['Model-1'].materials['SS316L For AM'].Density(dependencies=0,
0171:     distributionType=UNIFORM, fieldName='', table=((7980.0, 20.0), (7950.0,
0172:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0173:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
```

### N16 line 175 (mdb.models)

```text
0172:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0173:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
0174:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0175: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0176:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0177: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0178:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
```

### N16 line 177 (mdb.models)

```text
0174:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0175: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0176:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0177: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0178:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
0179:     0.3, 20.0), (202000000000.0, 0.3, 100.0), (194000000000.0, 0.3, 200.0), (
0180:     178000000000.0, 0.3, 400.0), (162000000000.0, 0.3, 600.0), (142000000000.0,
```

### N16 line 185 (mdb.models)

```text
0182:     15000000000.0, 0.3, 1375.0), (100000000.0, 0.3, 1400.0), (10000000.0, 0.3,
0183:     1723.0), (1000000.0, 0.3, 3000.0)), temperatureDependency=ON, type=
0184:     ISOTROPIC)
0185: mdb.models['Model-1'].materials['SS316L For AM'].Expansion(dependencies=0,
0186:     table=((1.48e-05, 20.0), (1.6e-05, 100.0), (1.68e-05, 200.0), (1.78e-05,
0187:     400.0), (1.87e-05, 600.0), (1.96e-05, 800.0), (2.02e-05, 1000.0), (
0188:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
```

### N16 line 191 (mdb.models)

```text
0188:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
0189:     1723.0), (2.2e-05, 3000.0)), temperatureDependency=ON, type=ISOTROPIC,
0190:     userSubroutine=OFF, zero=0.0)
0191: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0192:     1375.0, 1400.0), ))
0193: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0194:     '')
```

### N16 line 193 (mdb.models)

```text
0190:     userSubroutine=OFF, zero=0.0)
0191: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0192:     1375.0, 1400.0), ))
0193: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0194:     '')
0195: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0196:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
```

### N16 line 195 (mdb.models)

```text
0192:     1375.0, 1400.0), ))
0193: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0194:     '')
0195: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0196:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
0197:     numBackstresses=1, rate=OFF, scaleStress=None, staticRecovery=OFF,
0198:     strainRangeDependency=OFF, table=((580000000.0, 0.0, 20.0), (530000000.0,
```

### N16 line 204 (mdb.models)

```text
0201:     1000.0), (30000000.0, 0.0, 1200.0), (2000000.0, 0.0, 1375.0), (10000.0,
0202:     0.0, 1400.0), (5000.0, 0.0, 1723.0), (1000.0, 0.0, 3000.0)),
0203:     temperatureDependency=ON)
0204: mdb.models['Model-1'].materials['SS316L For AM'].SpecificHeat(dependencies=0,
0205:     law=CONSTANTVOLUME, table=((450.0, 20.0), (480.0, 100.0), (505.0, 200.0), (
0206:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0207:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
```

### N16 line 209 (mdb.models)

```text
0206:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0207:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
0208:     3000.0)), temperatureDependency=ON)
0209: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0210:     'property_section_all', thickness=None)
0211: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0212:     offsetField='', offsetType=MIDDLE_SURFACE, region=
```

### N16 line 211 (mdb.models)

```text
0208:     3000.0)), temperatureDependency=ON)
0209: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0210:     'property_section_all', thickness=None)
0211: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0212:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0213:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0214:     'property_section_all', thicknessAssignment=FROM_SECTION)
```

### N16 line 212 (region=)

```text
0209: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0210:     'property_section_all', thickness=None)
0211: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0212:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0213:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0214:     'property_section_all', thicknessAssignment=FROM_SECTION)
0215: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
```

### N16 line 213 (mdb.models)

```text
0210:     'property_section_all', thickness=None)
0211: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0212:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0213:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0214:     'property_section_all', thicknessAssignment=FROM_SECTION)
0215: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0216: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
```

### N16 line 215 (mdb.models)

```text
0212:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0213:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0214:     'property_section_all', thicknessAssignment=FROM_SECTION)
0215: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0216: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0217: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0218:     part=mdb.models['Model-1'].parts['part_plate'])
```

### N16 line 216 (mdb.models)

```text
0213:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0214:     'property_section_all', thicknessAssignment=FROM_SECTION)
0215: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0216: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0217: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0218:     part=mdb.models['Model-1'].parts['part_plate'])
0219: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
```

### N16 line 217 (mdb.models)

```text
0214:     'property_section_all', thicknessAssignment=FROM_SECTION)
0215: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0216: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0217: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0218:     part=mdb.models['Model-1'].parts['part_plate'])
0219: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0220:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
```

### N16 line 218 (mdb.models)

```text
0215: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0216: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0217: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0218:     part=mdb.models['Model-1'].parts['part_plate'])
0219: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0220:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0221:     nlgeom=ON, previous='Initial', timePeriod=0.2)
```

### N16 line 219 (mdb.models;CoupledTempDisplacementStep)

```text
0216: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0217: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0218:     part=mdb.models['Model-1'].parts['part_plate'])
0219: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0220:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0221:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0222: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
```

### N16 line 220 (step_scan_)

```text
0217: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0218:     part=mdb.models['Model-1'].parts['part_plate'])
0219: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0220:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0221:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0222: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0223:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
```

### N16 line 222 (mdb.models;CoupledTempDisplacementStep)

```text
0219: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0220:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0221:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0222: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0223:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0224:     previous='step_scan_00', timePeriod=3.4)
0225: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
```

### N16 line 223 (step_cool_)

```text
0220:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0221:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0222: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0223:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0224:     previous='step_scan_00', timePeriod=3.4)
0225: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0226:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
```

### N16 line 224 (step_scan_)

```text
0221:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0222: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0223:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0224:     previous='step_scan_00', timePeriod=3.4)
0225: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0226:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0227: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
```

### N16 line 225 (mdb.models)

```text
0222: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0223:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0224:     previous='step_scan_00', timePeriod=3.4)
0225: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0226:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0227: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0228:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
```

### N16 line 227 (mdb.models)

```text
0224:     previous='step_scan_00', timePeriod=3.4)
0225: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0226:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0227: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0228:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0229: mdb.models['Model-1'].FilmCondition(createStepName='step_cool_00', definition=
0230:     EMBEDDED_COEFF, filmCoeff=46.3, filmCoeffAmplitude='', name=
```

### N16 line 229 (step_cool_;mdb.models;createStepName)

```text
0226:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0227: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0228:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0229: mdb.models['Model-1'].FilmCondition(createStepName='step_cool_00', definition=
0230:     EMBEDDED_COEFF, filmCoeff=46.3, filmCoeffAmplitude='', name=
0231:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0232:     sinkFieldName='', sinkTemperature=20.0, surface=
```

### N16 line 233 (mdb.models)

```text
0230:     EMBEDDED_COEFF, filmCoeff=46.3, filmCoeffAmplitude='', name=
0231:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0232:     sinkFieldName='', sinkTemperature=20.0, surface=
0233:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0234: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_cool_00'
0235:     , 'step_scan_00')
0236: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_scan_00'
```

### N16 line 234 (step_cool_;mdb.models)

```text
0231:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0232:     sinkFieldName='', sinkTemperature=20.0, surface=
0233:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0234: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_cool_00'
0235:     , 'step_scan_00')
0236: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_scan_00'
0237:     , 'Initial')
```

### N16 line 235 (step_scan_)

```text
0232:     sinkFieldName='', sinkTemperature=20.0, surface=
0233:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0234: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_cool_00'
0235:     , 'step_scan_00')
0236: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_scan_00'
0237:     , 'Initial')
0238: #* ValueError: Film condition cannot be defined in the initial step.
```

### N16 line 236 (step_scan_;mdb.models)

```text
0233:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0234: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_cool_00'
0235:     , 'step_scan_00')
0236: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_scan_00'
0237:     , 'Initial')
0238: #* ValueError: Film condition cannot be defined in the initial step.
0239: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
```

### N16 line 239 (mdb.models)

```text
0236: mdb.models['Model-1'].interactions['film_external_cooling'].move('step_scan_00'
0237:     , 'Initial')
0238: #* ValueError: Film condition cannot be defined in the initial step.
0239: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0240:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0241:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0242:     radiationType=AMBIENT, surface=
```

### N16 line 240 (step_scan_;createStepName)

```text
0237:     , 'Initial')
0238: #* ValueError: Film condition cannot be defined in the initial step.
0239: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0240:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0241:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0242:     radiationType=AMBIENT, surface=
0243:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
```

### N16 line 243 (mdb.models)

```text
0240:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0241:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0242:     radiationType=AMBIENT, surface=
0243:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0244: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0245:     80000000000.0, name='load_body_hflux_00', region=
0246:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
```

### N16 line 244 (step_scan_;BodyHeatFlux;mdb.models;createStepName)

```text
0241:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0242:     radiationType=AMBIENT, surface=
0243:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0244: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0245:     80000000000.0, name='load_body_hflux_00', region=
0246:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0247: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N16 line 245 (load_body_hflux_;region=)

```text
0242:     radiationType=AMBIENT, surface=
0243:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0244: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0245:     80000000000.0, name='load_body_hflux_00', region=
0246:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0247: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0248:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
```

### N16 line 246 (set_body_heat_;mdb.models)

```text
0243:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0244: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0245:     80000000000.0, name='load_body_hflux_00', region=
0246:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0247: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0248:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0249:     region=Region(
```

### N16 line 247 (mdb.models;createStepName)

```text
0244: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0245:     80000000000.0, name='load_body_hflux_00', region=
0246:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0247: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0248:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0249:     region=Region(
0250:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N16 line 249 (region=;Region)

```text
0246:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0247: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0248:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0249:     region=Region(
0250:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0251:     mask=('[#0 #400000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0252: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N16 line 250 (mdb.models)

```text
0247: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0248:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0249:     region=Region(
0250:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0251:     mask=('[#0 #400000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0252: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0253:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
```

### N16 line 252 (mdb.models;createStepName)

```text
0249:     region=Region(
0250:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0251:     mask=('[#0 #400000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0252: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0253:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0254:     region=Region(
0255:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N16 line 254 (region=;Region)

```text
0251:     mask=('[#0 #400000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0252: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0253:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0254:     region=Region(
0255:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0256:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0257: mdb.models['Model-1'].Temperature(createStepName='Initial',
```

### N16 line 255 (mdb.models)

```text
0252: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0253:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0254:     region=Region(
0255:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0256:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0257: mdb.models['Model-1'].Temperature(createStepName='Initial',
0258:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
```

### N16 line 257 (mdb.models;createStepName)

```text
0254:     region=Region(
0255:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0256:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0257: mdb.models['Model-1'].Temperature(createStepName='Initial',
0258:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0259:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0260:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
```

### N16 line 259 (region=)

```text
0256:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0257: mdb.models['Model-1'].Temperature(createStepName='Initial',
0258:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0259:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0260:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0261: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0262:     minSizeFactor=0.1, size=0.0005)
```

### N16 line 260 (mdb.models)

```text
0257: mdb.models['Model-1'].Temperature(createStepName='Initial',
0258:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0259:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0260:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0261: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0262:     minSizeFactor=0.1, size=0.0005)
0263: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
```

### N16 line 261 (mdb.models)

```text
0258:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0259:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0260:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0261: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0262:     minSizeFactor=0.1, size=0.0005)
0263: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0264:     regions=
```

### N16 line 263 (mdb.models)

```text
0260:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0261: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0262:     minSizeFactor=0.1, size=0.0005)
0263: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0264:     regions=
0265:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0266:     '[#ffffffff #f ]', ), ), technique=STRUCTURED)
```

### N16 line 265 (mdb.models)

```text
0262:     minSizeFactor=0.1, size=0.0005)
0263: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0264:     regions=
0265:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0266:     '[#ffffffff #f ]', ), ), technique=STRUCTURED)
0267: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0268:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
```

### N16 line 267 (mdb.models)

```text
0264:     regions=
0265:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0266:     '[#ffffffff #f ]', ), ), technique=STRUCTURED)
0267: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0268:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0269:     elemLibrary=STANDARD)), regions=(
0270:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N16 line 270 (mdb.models)

```text
0267: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0268:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0269:     elemLibrary=STANDARD)), regions=(
0270:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0271:     '[#ffffffff #f ]', ), ), ))
0272: # Save by wuxia on 2026_06_10-21.56.26; build 2024 2023_09_21-20.55.25 RELr426 190762
0273: from part import *
```

### N16 line 286 (mdb.models)

```text
0283: from sketch import *
0284: from visualization import *
0285: from connectorBehavior import *
0286: mdb.models['Model-1'].rootAssembly.regenerate()
0287: # Save by wuxia on 2026_06_10-21.58.01; build 2024 2023_09_21-20.55.25 RELr426 190762
0288: from part import *
0289: from material import *
```

### N16 line 301 (step_cool_;load_body_hflux_;mdb.models;loads[;deactivate)

```text
0298: from sketch import *
0299: from visualization import *
0300: from connectorBehavior import *
0301: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0302: # Save by wuxia on 2026_06_10-23.01.27; build 2024 2023_09_21-20.55.25 RELr426 190762
```

### N24 line 15 (mdb.models)

```text
0012: from sketch import *
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.1)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
```

### N24 line 16 (mdb.models)

```text
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.1)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.052, 0.003))
```

### N24 line 18 (mdb.models)

```text
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.1)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.052, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
```

### N24 line 20 (mdb.models)

```text
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.052, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
```

### N24 line 22 (mdb.models)

```text
0019:     point2=(0.052, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__',
```

### N24 line 23 (mdb.models)

```text
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__',
0026:     sheetSize=0.104, transform=
```

### N24 line 24 (mdb.models)

```text
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__',
0026:     sheetSize=0.104, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
```

### N24 line 25 (mdb.models)

```text
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__',
0026:     sheetSize=0.104, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
```

### N24 line 27 (mdb.models)

```text
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__',
0026:     sheetSize=0.104, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.026, 0.0015,
0030:     0.0)))
```

### N24 line 28 (mdb.models)

```text
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.002, name='__profile__',
0026:     sheetSize=0.104, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.026, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
```

### N24 line 31 (mdb.models)

```text
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.026, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
```

### N24 line 33 (mdb.models)

```text
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.026, 0.0015))
```

### N24 line 34 (mdb.models)

```text
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.026, 0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N24 line 35 (mdb.models)

```text
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.026, 0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
```

### N24 line 37 (mdb.models)

```text
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0036:     point1=(-0.026, 0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N24 line 39 (mdb.models)

```text
0036:     point1=(-0.026, 0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
```

### N24 line 40 (mdb.models)

```text
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N24 line 41 (mdb.models)

```text
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
```

### N24 line 43 (mdb.models)

```text
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0045:     point1=(-0.026, -0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N24 line 44 (mdb.models)

```text
0041: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0042:     addUndoState=False, entity=
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0045:     point1=(-0.026, -0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
```

### N24 line 46 (mdb.models)

```text
0043:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0044: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0045:     point1=(-0.026, -0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N24 line 48 (mdb.models)

```text
0045:     point1=(-0.026, -0.0015))
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
```

### N24 line 49 (mdb.models)

```text
0046: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N24 line 50 (mdb.models)

```text
0047:     addUndoState=False, entity1=
0048:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
```

### N24 line 52 (mdb.models)

```text
0049:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0054:     point1=(-0.026, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N24 line 53 (mdb.models)

```text
0050: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0051:     addUndoState=False, entity=
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0054:     point1=(-0.026, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
```

### N24 line 55 (mdb.models)

```text
0052:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0053: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0054:     point1=(-0.026, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N24 line 57 (mdb.models)

```text
0054:     point1=(-0.026, -0.0015))
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N24 line 58 (mdb.models)

```text
0055: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0243921907842159,
```

### N24 line 59 (mdb.models)

```text
0056:     addUndoState=False, entity1=
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0243921907842159,
0062:     0.0015), point2=(-0.0243921907842159, -0.00150000000651926))
```

### N24 line 60 (mdb.models)

```text
0057:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0243921907842159,
0062:     0.0015), point2=(-0.0243921907842159, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
```

### N24 line 61 (mdb.models)

```text
0058:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0059: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0243921907842159,
0062:     0.0015), point2=(-0.0243921907842159, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N24 line 63 (mdb.models)

```text
0060:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0243921907842159,
0062:     0.0015), point2=(-0.0243921907842159, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
```

### N24 line 64 (mdb.models)

```text
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0243921907842159,
0062:     0.0015), point2=(-0.0243921907842159, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
```

### N24 line 65 (mdb.models)

```text
0062:     0.0015), point2=(-0.0243921907842159, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N24 line 67 (mdb.models)

```text
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
```

### N24 line 68 (mdb.models)

```text
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
```

### N24 line 69 (mdb.models)

```text
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
```

### N24 line 71 (mdb.models)

```text
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
```

### N24 line 72 (mdb.models)

```text
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
```

### N24 line 73 (mdb.models)

```text
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
```

### N24 line 75 (mdb.models)

```text
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], entity2=
```

### N24 line 76 (mdb.models)

```text
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], textPoint=(
```

### N24 line 77 (mdb.models)

```text
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], textPoint=(
0080:     -0.0248637268543243, 0.00379133621603251), value=0.002)
```

### N24 line 78 (mdb.models)

```text
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], textPoint=(
0080:     -0.0248637268543243, 0.00379133621603251), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
```

### N24 line 79 (mdb.models)

```text
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], textPoint=(
0080:     -0.0248637268543243, 0.00379133621603251), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
```

### N24 line 81 (mdb.models)

```text
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[8], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], textPoint=(
0080:     -0.0248637268543243, 0.00379133621603251), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=25, number2=1, spacing1=0.002, spacing2=0.0104, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.026,
```

### N24 line 82 (mdb.models)

```text
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], textPoint=(
0080:     -0.0248637268543243, 0.00379133621603251), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=25, number2=1, spacing1=0.002, spacing2=0.0104, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.026,
0085:     0.000673049142584205), point2=(0.0260000000353903, 0.000673049142584205))
```

### N24 line 84 (mdb.models)

```text
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=25, number2=1, spacing1=0.002, spacing2=0.0104, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.026,
0085:     0.000673049142584205), point2=(0.0260000000353903, 0.000673049142584205))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
```

### N24 line 86 (mdb.models)

```text
0083:     ), number1=25, number2=1, spacing1=0.002, spacing2=0.0104, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.026,
0085:     0.000673049142584205), point2=(0.0260000000353903, 0.000673049142584205))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
```

### N24 line 88 (mdb.models)

```text
0085:     0.000673049142584205), point2=(0.0260000000353903, 0.000673049142584205))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
```

### N24 line 89 (mdb.models)

```text
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
```

### N24 line 91 (mdb.models)

```text
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
```

### N24 line 92 (mdb.models)

```text
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[54], entity2=
```

### N24 line 93 (mdb.models)

```text
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[54], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
```

### N24 line 95 (mdb.models)

```text
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[34])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[54], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
```

### N24 line 96 (mdb.models)

```text
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[54], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[55], entity2=
```

### N24 line 97 (mdb.models)

```text
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[54], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[55], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
```

### N24 line 99 (mdb.models)

```text
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[55], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[34], entity2=
```

### N24 line 100 (mdb.models)

```text
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[55], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[34], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
```

### N24 line 101 (mdb.models)

```text
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[55], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[34], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0281608956158161, -0.000651412630453706), value=0.002)
```

### N24 line 102 (mdb.models)

```text
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[55], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[34], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0281608956158161, -0.000651412630453706), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
```

### N24 line 103 (mdb.models)

```text
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[34], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0281608956158161, -0.000651412630453706), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 105 (mdb.models)

```text
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[34], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0281608956158161, -0.000651412630453706), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
```

### N24 line 106 (mdb.models)

```text
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[2], textPoint=(
0104:     -0.0281608956158161, -0.000651412630453706), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 107 (mdb.models)

```text
0104:     -0.0281608956158161, -0.000651412630453706), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 108 (mdb.models)

```text
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #fffff ]', ), ), name='section_all')
```

### N24 line 109 (mdb.models)

```text
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 110 (mdb.models)

```text
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 112 (mdb.models)

```text
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 113 (mdb.models)

```text
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 114 (set_body_heat_)

```text
0111:     '[#ffffffff #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0 #8000 ]', ), ), name='set_body_heat_01')
```

### N24 line 115 (mdb.models)

```text
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 116 (mdb.models)

```text
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 117 (set_body_heat_)

```text
0114:     '[#0 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0 #2000 ]', ), ), name='set_body_heat_02')
```

### N24 line 118 (mdb.models)

```text
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 119 (mdb.models)

```text
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 120 (set_body_heat_)

```text
0117:     '[#0 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0 #800 ]', ), ), name='set_body_heat_03')
```

### N24 line 121 (mdb.models)

```text
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 122 (mdb.models)

```text
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 123 (set_body_heat_)

```text
0120:     '[#0 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0 #200 ]', ), ), name='set_body_heat_04')
```

### N24 line 124 (mdb.models)

```text
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 125 (mdb.models)

```text
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 126 (set_body_heat_)

```text
0123:     '[#0 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0 #80 ]', ), ), name='set_body_heat_05')
```

### N24 line 127 (mdb.models)

```text
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 128 (mdb.models)

```text
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 129 (set_body_heat_)

```text
0126:     '[#0 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0 #20 ]', ), ), name='set_body_heat_06')
```

### N24 line 130 (mdb.models)

```text
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 131 (mdb.models)

```text
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 132 (set_body_heat_)

```text
0129:     '[#0 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0 #8 ]', ), ), name='set_body_heat_07')
```

### N24 line 133 (mdb.models)

```text
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 134 (mdb.models)

```text
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 135 (set_body_heat_)

```text
0132:     '[#0 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0 #2 ]', ), ), name='set_body_heat_08')
```

### N24 line 136 (mdb.models)

```text
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 137 (mdb.models)

```text
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 138 (set_body_heat_)

```text
0135:     '[#0 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#80000000 ]', ), ), name='set_body_heat_09')
```

### N24 line 139 (mdb.models)

```text
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 140 (mdb.models)

```text
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 141 (set_body_heat_)

```text
0138:     '[#0 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#20000000 ]', ), ), name='set_body_heat_10')
```

### N24 line 142 (mdb.models)

```text
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 143 (mdb.models)

```text
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 144 (set_body_heat_)

```text
0141:     '[#80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#8000000 ]', ), ), name='set_body_heat_11')
```

### N24 line 145 (mdb.models)

```text
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 146 (mdb.models)

```text
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 147 (set_body_heat_)

```text
0144:     '[#20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#2000000 ]', ), ), name='set_body_heat_12')
```

### N24 line 148 (mdb.models)

```text
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 149 (mdb.models)

```text
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 150 (set_body_heat_)

```text
0147:     '[#8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#800000 ]', ), ), name='set_body_heat_13')
```

### N24 line 151 (mdb.models)

```text
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 152 (mdb.models)

```text
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 153 (set_body_heat_)

```text
0150:     '[#2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#200000 ]', ), ), name='set_body_heat_14')
```

### N24 line 154 (mdb.models)

```text
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 155 (mdb.models)

```text
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 156 (set_body_heat_)

```text
0153:     '[#800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#80000 ]', ), ), name='set_body_heat_15')
```

### N24 line 157 (mdb.models)

```text
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 158 (mdb.models)

```text
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 159 (set_body_heat_)

```text
0156:     '[#200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#20000 ]', ), ), name='set_body_heat_16')
```

### N24 line 160 (mdb.models)

```text
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 161 (mdb.models)

```text
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 162 (set_body_heat_)

```text
0159:     '[#80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#8000 ]', ), ), name='set_body_heat_17')
```

### N24 line 163 (mdb.models)

```text
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 164 (mdb.models)

```text
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 165 (set_body_heat_)

```text
0162:     '[#20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#2000 ]', ), ), name='set_body_heat_18')
```

### N24 line 166 (mdb.models)

```text
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 167 (mdb.models)

```text
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 168 (set_body_heat_)

```text
0165:     '[#8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#800 ]', ), ), name='set_body_heat_19')
```

### N24 line 169 (mdb.models)

```text
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 170 (mdb.models)

```text
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 171 (set_body_heat_)

```text
0168:     '[#2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#200 ]', ), ), name='set_body_heat_20')
```

### N24 line 172 (mdb.models)

```text
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 173 (mdb.models)

```text
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 174 (set_body_heat_)

```text
0171:     '[#800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#80 ]', ), ), name='set_body_heat_21')
```

### N24 line 175 (mdb.models)

```text
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 176 (mdb.models)

```text
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 177 (set_body_heat_)

```text
0174:     '[#200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#20 ]', ), ), name='set_body_heat_22')
```

### N24 line 178 (mdb.models)

```text
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N24 line 179 (mdb.models)

```text
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 180 (set_body_heat_)

```text
0177:     '[#80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#8 ]', ), ), name='set_body_heat_23')
```

### N24 line 181 (mdb.models)

```text
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
```

### N24 line 182 (mdb.models)

```text
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0185:     side1Edges=
```

### N24 line 183 (set_body_heat_)

```text
0180:     '[#20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0185:     side1Edges=
0186:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
```

### N24 line 184 (mdb.models)

```text
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0185:     side1Edges=
0186:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0187:     '[#94a54966 #a5294a52 #294a5294 #ca5294a5 #e ]', ), ))
```

### N24 line 186 (mdb.models)

```text
0183:     '[#8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0185:     side1Edges=
0186:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0187:     '[#94a54966 #a5294a52 #294a5294 #ca5294a5 #e ]', ), ))
0188: mdb.models['Model-1'].Material(name='SS316L For AM')
0189: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
```

### N24 line 188 (mdb.models)

```text
0185:     side1Edges=
0186:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0187:     '[#94a54966 #a5294a52 #294a5294 #ca5294a5 #e ]', ), ))
0188: mdb.models['Model-1'].Material(name='SS316L For AM')
0189: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0190:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0191:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
```

### N24 line 189 (mdb.models)

```text
0186:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0187:     '[#94a54966 #a5294a52 #294a5294 #ca5294a5 #e ]', ), ))
0188: mdb.models['Model-1'].Material(name='SS316L For AM')
0189: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0190:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0191:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0192:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
```

### N24 line 194 (mdb.models)

```text
0191:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0192:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
0193:     type=ISOTROPIC)
0194: mdb.models['Model-1'].materials['SS316L For AM'].Density(dependencies=0,
0195:     distributionType=UNIFORM, fieldName='', table=((7980.0, 20.0), (7950.0,
0196:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0197:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
```

### N24 line 199 (mdb.models)

```text
0196:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0197:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
0198:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0199: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0200:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0201: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0202:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
```

### N24 line 201 (mdb.models)

```text
0198:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0199: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0200:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0201: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0202:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
0203:     0.3, 20.0), (202000000000.0, 0.3, 100.0), (194000000000.0, 0.3, 200.0), (
0204:     178000000000.0, 0.3, 400.0), (162000000000.0, 0.3, 600.0), (142000000000.0,
```

### N24 line 209 (mdb.models)

```text
0206:     15000000000.0, 0.3, 1375.0), (100000000.0, 0.3, 1400.0), (10000000.0, 0.3,
0207:     1723.0), (1000000.0, 0.3, 3000.0)), temperatureDependency=ON, type=
0208:     ISOTROPIC)
0209: mdb.models['Model-1'].materials['SS316L For AM'].Expansion(dependencies=0,
0210:     table=((1.48e-05, 20.0), (1.6e-05, 100.0), (1.68e-05, 200.0), (1.78e-05,
0211:     400.0), (1.87e-05, 600.0), (1.96e-05, 800.0), (2.02e-05, 1000.0), (
0212:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
```

### N24 line 215 (mdb.models)

```text
0212:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
0213:     1723.0), (2.2e-05, 3000.0)), temperatureDependency=ON, type=ISOTROPIC,
0214:     userSubroutine=OFF, zero=0.0)
0215: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0216:     1375.0, 1400.0), ))
0217: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0218:     '')
```

### N24 line 217 (mdb.models)

```text
0214:     userSubroutine=OFF, zero=0.0)
0215: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0216:     1375.0, 1400.0), ))
0217: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0218:     '')
0219: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0220:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
```

### N24 line 219 (mdb.models)

```text
0216:     1375.0, 1400.0), ))
0217: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0218:     '')
0219: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0220:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
0221:     numBackstresses=1, rate=OFF, scaleStress=None, staticRecovery=OFF,
0222:     strainRangeDependency=OFF, table=((580000000.0, 0.0, 20.0), (530000000.0,
```

### N24 line 228 (mdb.models)

```text
0225:     1000.0), (30000000.0, 0.0, 1200.0), (2000000.0, 0.0, 1375.0), (10000.0,
0226:     0.0, 1400.0), (5000.0, 0.0, 1723.0), (1000.0, 0.0, 3000.0)),
0227:     temperatureDependency=ON)
0228: mdb.models['Model-1'].materials['SS316L For AM'].SpecificHeat(dependencies=0,
0229:     law=CONSTANTVOLUME, table=((450.0, 20.0), (480.0, 100.0), (505.0, 200.0), (
0230:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0231:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
```

### N24 line 233 (mdb.models)

```text
0230:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0231:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
0232:     3000.0)), temperatureDependency=ON)
0233: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0234:     'property_section_all', thickness=None)
0235: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0236:     offsetField='', offsetType=MIDDLE_SURFACE, region=
```

### N24 line 235 (mdb.models)

```text
0232:     3000.0)), temperatureDependency=ON)
0233: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0234:     'property_section_all', thickness=None)
0235: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0236:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0237:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0238:     'property_section_all', thicknessAssignment=FROM_SECTION)
```

### N24 line 236 (region=)

```text
0233: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0234:     'property_section_all', thickness=None)
0235: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0236:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0237:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0238:     'property_section_all', thicknessAssignment=FROM_SECTION)
0239: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
```

### N24 line 237 (mdb.models)

```text
0234:     'property_section_all', thickness=None)
0235: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0236:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0237:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0238:     'property_section_all', thicknessAssignment=FROM_SECTION)
0239: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0240: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
```

### N24 line 239 (mdb.models)

```text
0236:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0237:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0238:     'property_section_all', thicknessAssignment=FROM_SECTION)
0239: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0240: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0241: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0242:     part=mdb.models['Model-1'].parts['part_plate'])
```

### N24 line 240 (mdb.models)

```text
0237:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0238:     'property_section_all', thicknessAssignment=FROM_SECTION)
0239: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0240: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0241: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0242:     part=mdb.models['Model-1'].parts['part_plate'])
0243: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
```

### N24 line 241 (mdb.models)

```text
0238:     'property_section_all', thicknessAssignment=FROM_SECTION)
0239: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0240: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0241: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0242:     part=mdb.models['Model-1'].parts['part_plate'])
0243: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0244:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
```

### N24 line 242 (mdb.models)

```text
0239: mdb.models['Model-1'].setValues(absoluteZero=-273.15, stefanBoltzmann=5.67e-08)
0240: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0241: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0242:     part=mdb.models['Model-1'].parts['part_plate'])
0243: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0244:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0245:     nlgeom=ON, previous='Initial', timePeriod=0.2)
```

### N24 line 243 (mdb.models;CoupledTempDisplacementStep)

```text
0240: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0241: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0242:     part=mdb.models['Model-1'].parts['part_plate'])
0243: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0244:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0245:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0246: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
```

### N24 line 244 (step_scan_)

```text
0241: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0242:     part=mdb.models['Model-1'].parts['part_plate'])
0243: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0244:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0245:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0246: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0247:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
```

### N24 line 246 (mdb.models;CoupledTempDisplacementStep)

```text
0243: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0244:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0245:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0246: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0247:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0248:     previous='step_scan_00', timePeriod=3.4)
0249: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
```

### N24 line 247 (step_cool_)

```text
0244:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0245:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0246: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0247:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0248:     previous='step_scan_00', timePeriod=3.4)
0249: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0250:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
```

### N24 line 248 (step_scan_)

```text
0245:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0246: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0247:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0248:     previous='step_scan_00', timePeriod=3.4)
0249: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0250:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0251: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
```

### N24 line 249 (mdb.models)

```text
0246: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.01
0247:     , maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0248:     previous='step_scan_00', timePeriod=3.4)
0249: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0250:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0251: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0252:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
```

### N24 line 251 (mdb.models)

```text
0248:     previous='step_scan_00', timePeriod=3.4)
0249: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0250:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0251: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0252:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0253: mdb.models['Model-1'].FilmCondition(createStepName='step_scan_00', definition=
0254:     EMBEDDED_COEFF, filmCoeff=46.5, filmCoeffAmplitude='', name=
```

### N24 line 253 (step_scan_;mdb.models;createStepName)

```text
0250:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0251: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0252:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0253: mdb.models['Model-1'].FilmCondition(createStepName='step_scan_00', definition=
0254:     EMBEDDED_COEFF, filmCoeff=46.5, filmCoeffAmplitude='', name=
0255:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0256:     sinkFieldName='', sinkTemperature=20.0, surface=
```

### N24 line 257 (mdb.models)

```text
0254:     EMBEDDED_COEFF, filmCoeff=46.5, filmCoeffAmplitude='', name=
0255:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0256:     sinkFieldName='', sinkTemperature=20.0, surface=
0257:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0258: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0259:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0260:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
```

### N24 line 258 (mdb.models)

```text
0255:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0256:     sinkFieldName='', sinkTemperature=20.0, surface=
0257:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0258: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0259:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0260:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0261:     radiationType=AMBIENT, surface=
```

### N24 line 259 (step_scan_;createStepName)

```text
0256:     sinkFieldName='', sinkTemperature=20.0, surface=
0257:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0258: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0259:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0260:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0261:     radiationType=AMBIENT, surface=
0262:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
```

### N24 line 262 (mdb.models)

```text
0259:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0260:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0261:     radiationType=AMBIENT, surface=
0262:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0263: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0264:     80000000000.0, name='load_body_hflux_00', region=
0265:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
```

### N24 line 263 (step_scan_;BodyHeatFlux;mdb.models;createStepName)

```text
0260:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0261:     radiationType=AMBIENT, surface=
0262:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0263: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0264:     80000000000.0, name='load_body_hflux_00', region=
0265:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0266: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N24 line 264 (load_body_hflux_;region=)

```text
0261:     radiationType=AMBIENT, surface=
0262:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0263: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0264:     80000000000.0, name='load_body_hflux_00', region=
0265:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0266: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0267:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
```

### N24 line 265 (set_body_heat_;mdb.models)

```text
0262:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0263: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0264:     80000000000.0, name='load_body_hflux_00', region=
0265:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0266: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0267:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0268:     region=Region(
```

### N24 line 266 (mdb.models;createStepName)

```text
0263: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0264:     80000000000.0, name='load_body_hflux_00', region=
0265:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0266: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0267:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0268:     region=Region(
0269:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N24 line 268 (region=;Region)

```text
0265:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0266: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0267:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0268:     region=Region(
0269:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0270:     mask=('[#0:2 #4000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0271: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N24 line 269 (mdb.models)

```text
0266: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0267:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0268:     region=Region(
0269:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0270:     mask=('[#0:2 #4000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0271: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0272:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
```

### N24 line 271 (mdb.models;createStepName)

```text
0268:     region=Region(
0269:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0270:     mask=('[#0:2 #4000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0271: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0272:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0273:     region=Region(
0274:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N24 line 273 (region=;Region)

```text
0270:     mask=('[#0:2 #4000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0271: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0272:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0273:     region=Region(
0274:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0275:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0276: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
```

### N24 line 274 (mdb.models)

```text
0271: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0272:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0273:     region=Region(
0274:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0275:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0276: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0277: mdb.models['Model-1'].Temperature(createStepName='Initial',
```

### N24 line 276 (step_cool_;load_body_hflux_;mdb.models;loads[;deactivate)

```text
0273:     region=Region(
0274:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0275:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0276: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0277: mdb.models['Model-1'].Temperature(createStepName='Initial',
0278:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0279:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
```

### N24 line 277 (mdb.models;createStepName)

```text
0274:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0275:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0276: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0277: mdb.models['Model-1'].Temperature(createStepName='Initial',
0278:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0279:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0280:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
```

### N24 line 279 (region=)

```text
0276: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0277: mdb.models['Model-1'].Temperature(createStepName='Initial',
0278:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0279:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0280:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0281: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0282:     minSizeFactor=0.1, size=0.0005)
```

### N24 line 280 (mdb.models)

```text
0277: mdb.models['Model-1'].Temperature(createStepName='Initial',
0278:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0279:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0280:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0281: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0282:     minSizeFactor=0.1, size=0.0005)
0283: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
```

### N24 line 281 (mdb.models)

```text
0278:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0279:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0280:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0281: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0282:     minSizeFactor=0.1, size=0.0005)
0283: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0284:     regions=
```

### N24 line 283 (mdb.models)

```text
0280:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0281: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0282:     minSizeFactor=0.1, size=0.0005)
0283: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0284:     regions=
0285:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0286:     '[#ffffffff #fffff ]', ), ), technique=STRUCTURED)
```

### N24 line 285 (mdb.models)

```text
0282:     minSizeFactor=0.1, size=0.0005)
0283: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0284:     regions=
0285:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0286:     '[#ffffffff #fffff ]', ), ), technique=STRUCTURED)
0287: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0288:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
```

### N24 line 287 (mdb.models)

```text
0284:     regions=
0285:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0286:     '[#ffffffff #fffff ]', ), ), technique=STRUCTURED)
0287: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0288:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0289:     elemLibrary=STANDARD)), regions=(
0290:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N24 line 290 (mdb.models)

```text
0287: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0288:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0289:     elemLibrary=STANDARD)), regions=(
0290:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0291:     '[#ffffffff #fffff ]', ), ), ))
0292: # Save by wuxia on 2026_06_10-22.24.04; build 2024 2023_09_21-20.55.25 RELr426 190762
0293: from part import *
```

### N24 line 306 (mdb.models)

```text
0303: from sketch import *
0304: from visualization import *
0305: from connectorBehavior import *
0306: mdb.models['Model-1'].rootAssembly.regenerate()
0307: # Save by wuxia on 2026_06_10-22.24.45; build 2024 2023_09_21-20.55.25 RELr426 190762
```

### N40 line 15 (mdb.models)

```text
0012: from sketch import *
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.2)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
```

### N40 line 16 (mdb.models)

```text
0013: from visualization import *
0014: from connectorBehavior import *
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.2)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.084, 0.003))
```

### N40 line 18 (mdb.models)

```text
0015: mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=0.2)
0016: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.084, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
```

### N40 line 20 (mdb.models)

```text
0017:     decimalPlaces=3)
0018: mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(0.0, 0.0),
0019:     point2=(0.084, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
```

### N40 line 22 (mdb.models)

```text
0019:     point2=(0.084, 0.003))
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.004, name='__profile__',
```

### N40 line 23 (mdb.models)

```text
0020: mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='part_plate',
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.004, name='__profile__',
0026:     sheetSize=0.168, transform=
```

### N40 line 24 (mdb.models)

```text
0021:     type=DEFORMABLE_BODY)
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.004, name='__profile__',
0026:     sheetSize=0.168, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
```

### N40 line 25 (mdb.models)

```text
0022: mdb.models['Model-1'].parts['part_plate'].BaseShell(sketch=
0023:     mdb.models['Model-1'].sketches['__profile__'])
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.004, name='__profile__',
0026:     sheetSize=0.168, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
```

### N40 line 27 (mdb.models)

```text
0024: del mdb.models['Model-1'].sketches['__profile__']
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.004, name='__profile__',
0026:     sheetSize=0.168, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.042, 0.0015,
0030:     0.0)))
```

### N40 line 28 (mdb.models)

```text
0025: mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.004, name='__profile__',
0026:     sheetSize=0.168, transform=
0027:     mdb.models['Model-1'].parts['part_plate'].MakeSketchTransform(
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.042, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
```

### N40 line 31 (mdb.models)

```text
0028:     sketchPlane=mdb.models['Model-1'].parts['part_plate'].faces[0],
0029:     sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.042, 0.0015,
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
```

### N40 line 33 (mdb.models)

```text
0030:     0.0)))
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.042, -0.0015))
```

### N40 line 34 (mdb.models)

```text
0031: mdb.models['Model-1'].sketches['__profile__'].sketchOptions.setValues(
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.042, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N40 line 35 (mdb.models)

```text
0032:     decimalPlaces=3)
0033: mdb.models['Model-1'].parts['part_plate'].projectReferencesOntoSketch(filter=
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.042, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
```

### N40 line 37 (mdb.models)

```text
0034:     COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])
0035: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=90.0,
0036:     point1=(-0.042, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N40 line 39 (mdb.models)

```text
0036:     point1=(-0.042, -0.0015))
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
```

### N40 line 40 (mdb.models)

```text
0037: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
```

### N40 line 41 (mdb.models)

```text
0038:     addUndoState=False, entity1=
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.042, -0.0015))
```

### N40 line 42 (mdb.models)

```text
0039:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.042, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N40 line 43 (mdb.models)

```text
0040:     mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0041: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.042, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
```

### N40 line 45 (mdb.models)

```text
0042:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[6])
0043: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0044:     point1=(-0.042, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N40 line 47 (mdb.models)

```text
0044:     point1=(-0.042, -0.0015))
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
```

### N40 line 48 (mdb.models)

```text
0045: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
```

### N40 line 49 (mdb.models)

```text
0046:     addUndoState=False, entity1=
0047:     mdb.models['Model-1'].sketches['__profile__'].vertices[0], entity2=
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
```

### N40 line 51 (mdb.models)

```text
0048:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0053:     point1=(-0.042, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
```

### N40 line 52 (mdb.models)

```text
0049: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0050:     addUndoState=False, entity=
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0053:     point1=(-0.042, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
```

### N40 line 54 (mdb.models)

```text
0051:     mdb.models['Model-1'].sketches['__profile__'].geometry[7])
0052: mdb.models['Model-1'].sketches['__profile__'].ConstructionLine(angle=0.0,
0053:     point1=(-0.042, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N40 line 56 (mdb.models)

```text
0053:     point1=(-0.042, 0.0015))
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
```

### N40 line 57 (mdb.models)

```text
0054: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
```

### N40 line 58 (mdb.models)

```text
0055:     addUndoState=False, entity1=
0056:     mdb.models['Model-1'].sketches['__profile__'].vertices[3], entity2=
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0404501227736473,
```

### N40 line 60 (mdb.models)

```text
0057:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0404501227736473,
0062:     0.0015), point2=(-0.0404501227736473, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
```

### N40 line 61 (mdb.models)

```text
0058: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0059:     addUndoState=False, entity=
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0404501227736473,
0062:     0.0015), point2=(-0.0404501227736473, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N40 line 63 (mdb.models)

```text
0060:     mdb.models['Model-1'].sketches['__profile__'].geometry[8])
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0404501227736473,
0062:     0.0015), point2=(-0.0404501227736473, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
```

### N40 line 64 (mdb.models)

```text
0061: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.0404501227736473,
0062:     0.0015), point2=(-0.0404501227736473, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
```

### N40 line 65 (mdb.models)

```text
0062:     0.0015), point2=(-0.0404501227736473, -0.00150000000651926))
0063: mdb.models['Model-1'].sketches['__profile__'].VerticalConstraint(addUndoState=
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
```

### N40 line 67 (mdb.models)

```text
0064:     False, entity=mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
```

### N40 line 68 (mdb.models)

```text
0065: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
```

### N40 line 69 (mdb.models)

```text
0066:     addUndoState=False, entity1=
0067:     mdb.models['Model-1'].sketches['__profile__'].geometry[4], entity2=
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
```

### N40 line 71 (mdb.models)

```text
0068:     mdb.models['Model-1'].sketches['__profile__'].geometry[9])
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
```

### N40 line 72 (mdb.models)

```text
0069: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
```

### N40 line 73 (mdb.models)

```text
0070:     addUndoState=False, entity1=
0071:     mdb.models['Model-1'].sketches['__profile__'].vertices[4], entity2=
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
```

### N40 line 75 (mdb.models)

```text
0072:     mdb.models['Model-1'].sketches['__profile__'].geometry[4])
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
```

### N40 line 76 (mdb.models)

```text
0073: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
```

### N40 line 77 (mdb.models)

```text
0074:     addUndoState=False, entity1=
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0411800971329212, -0.00590233293920755), value=0.002)
```

### N40 line 78 (mdb.models)

```text
0075:     mdb.models['Model-1'].sketches['__profile__'].vertices[5], entity2=
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0411800971329212, -0.00590233293920755), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
```

### N40 line 79 (mdb.models)

```text
0076:     mdb.models['Model-1'].sketches['__profile__'].geometry[2])
0077: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0411800971329212, -0.00590233293920755), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
```

### N40 line 81 (mdb.models)

```text
0078:     mdb.models['Model-1'].sketches['__profile__'].geometry[9], entity2=
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0411800971329212, -0.00590233293920755), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=41, number2=1, spacing1=0.002, spacing2=0.0168, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.042,
```

### N40 line 82 (mdb.models)

```text
0079:     mdb.models['Model-1'].sketches['__profile__'].geometry[6], textPoint=(
0080:     -0.0411800971329212, -0.00590233293920755), value=0.002)
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=41, number2=1, spacing1=0.002, spacing2=0.0168, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.042,
0085:     0.000539154060184956), point2=(0.0419999999497086, 0.000539154060184956))
```

### N40 line 84 (mdb.models)

```text
0081: mdb.models['Model-1'].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
0082:     90.0, geomList=(mdb.models['Model-1'].sketches['__profile__'].geometry[9],
0083:     ), number1=41, number2=1, spacing1=0.002, spacing2=0.0168, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.042,
0085:     0.000539154060184956), point2=(0.0419999999497086, 0.000539154060184956))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
```

### N40 line 86 (mdb.models)

```text
0083:     ), number1=41, number2=1, spacing1=0.002, spacing2=0.0168, vertexList=())
0084: mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-0.042,
0085:     0.000539154060184956), point2=(0.0419999999497086, 0.000539154060184956))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
```

### N40 line 88 (mdb.models)

```text
0085:     0.000539154060184956), point2=(0.0419999999497086, 0.000539154060184956))
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
```

### N40 line 89 (mdb.models)

```text
0086: mdb.models['Model-1'].sketches['__profile__'].HorizontalConstraint(
0087:     addUndoState=False, entity=
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
```

### N40 line 91 (mdb.models)

```text
0088:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
```

### N40 line 92 (mdb.models)

```text
0089: mdb.models['Model-1'].sketches['__profile__'].PerpendicularConstraint(
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[86], entity2=
```

### N40 line 93 (mdb.models)

```text
0090:     addUndoState=False, entity1=
0091:     mdb.models['Model-1'].sketches['__profile__'].geometry[5], entity2=
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[86], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
```

### N40 line 95 (mdb.models)

```text
0092:     mdb.models['Model-1'].sketches['__profile__'].geometry[50])
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[86], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
```

### N40 line 96 (mdb.models)

```text
0093: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[86], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[87], entity2=
```

### N40 line 97 (mdb.models)

```text
0094:     addUndoState=False, entity1=
0095:     mdb.models['Model-1'].sketches['__profile__'].vertices[86], entity2=
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[87], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
```

### N40 line 99 (mdb.models)

```text
0096:     mdb.models['Model-1'].sketches['__profile__'].geometry[5])
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[87], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[50], entity2=
```

### N40 line 100 (mdb.models)

```text
0097: mdb.models['Model-1'].sketches['__profile__'].CoincidentConstraint(
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[87], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[50], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
```

### N40 line 101 (mdb.models)

```text
0098:     addUndoState=False, entity1=
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[87], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[50], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0458642175495625, -0.000261133756488562), value=0.002)
```

### N40 line 102 (mdb.models)

```text
0099:     mdb.models['Model-1'].sketches['__profile__'].vertices[87], entity2=
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[50], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0458642175495625, -0.000261133756488562), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
```

### N40 line 103 (mdb.models)

```text
0100:     mdb.models['Model-1'].sketches['__profile__'].geometry[3])
0101: mdb.models['Model-1'].sketches['__profile__'].DistanceDimension(entity1=
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[50], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0458642175495625, -0.000261133756488562), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 105 (mdb.models)

```text
0102:     mdb.models['Model-1'].sketches['__profile__'].geometry[50], entity2=
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0458642175495625, -0.000261133756488562), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
```

### N40 line 106 (mdb.models)

```text
0103:     mdb.models['Model-1'].sketches['__profile__'].geometry[7], textPoint=(
0104:     -0.0458642175495625, -0.000261133756488562), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 107 (mdb.models)

```text
0104:     -0.0458642175495625, -0.000261133756488562), value=0.002)
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 108 (mdb.models)

```text
0105: mdb.models['Model-1'].parts['part_plate'].PartitionFaceBySketch(faces=
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff:2 #fffff ]', ), ), name='section_all')
```

### N40 line 109 (mdb.models)

```text
0106:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff:2 #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 110 (mdb.models)

```text
0107:     '[#1 ]', ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])
0108: del mdb.models['Model-1'].sketches['__profile__']
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff:2 #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 112 (mdb.models)

```text
0109: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff:2 #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0:2 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 113 (mdb.models)

```text
0110:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0111:     '[#ffffffff:2 #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0:2 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 114 (set_body_heat_)

```text
0111:     '[#ffffffff:2 #fffff ]', ), ), name='section_all')
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0:2 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0:2 #8000 ]', ), ), name='set_body_heat_01')
```

### N40 line 115 (mdb.models)

```text
0112: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0:2 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0:2 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 116 (mdb.models)

```text
0113:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0114:     '[#0:2 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0:2 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 117 (set_body_heat_)

```text
0114:     '[#0:2 #40000 ]', ), ), name='set_body_heat_00')
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0:2 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0:2 #2000 ]', ), ), name='set_body_heat_02')
```

### N40 line 118 (mdb.models)

```text
0115: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0:2 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0:2 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 119 (mdb.models)

```text
0116:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0117:     '[#0:2 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0:2 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 120 (set_body_heat_)

```text
0117:     '[#0:2 #8000 ]', ), ), name='set_body_heat_01')
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0:2 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0:2 #800 ]', ), ), name='set_body_heat_03')
```

### N40 line 121 (mdb.models)

```text
0118: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0:2 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0:2 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 122 (mdb.models)

```text
0119:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0120:     '[#0:2 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0:2 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 123 (set_body_heat_)

```text
0120:     '[#0:2 #2000 ]', ), ), name='set_body_heat_02')
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0:2 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0:2 #200 ]', ), ), name='set_body_heat_04')
```

### N40 line 124 (mdb.models)

```text
0121: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0:2 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0:2 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 125 (mdb.models)

```text
0122:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0123:     '[#0:2 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0:2 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 126 (set_body_heat_)

```text
0123:     '[#0:2 #800 ]', ), ), name='set_body_heat_03')
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0:2 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0:2 #80 ]', ), ), name='set_body_heat_05')
```

### N40 line 127 (mdb.models)

```text
0124: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0:2 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0:2 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 128 (mdb.models)

```text
0125:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0126:     '[#0:2 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0:2 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 129 (set_body_heat_)

```text
0126:     '[#0:2 #200 ]', ), ), name='set_body_heat_04')
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0:2 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0:2 #20 ]', ), ), name='set_body_heat_06')
```

### N40 line 130 (mdb.models)

```text
0127: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0:2 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0:2 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 131 (mdb.models)

```text
0128:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0129:     '[#0:2 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0:2 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 132 (set_body_heat_)

```text
0129:     '[#0:2 #80 ]', ), ), name='set_body_heat_05')
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0:2 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0:2 #8 ]', ), ), name='set_body_heat_07')
```

### N40 line 133 (mdb.models)

```text
0130: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0:2 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0:2 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 134 (mdb.models)

```text
0131:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0132:     '[#0:2 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0:2 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 135 (set_body_heat_)

```text
0132:     '[#0:2 #20 ]', ), ), name='set_body_heat_06')
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0:2 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0:2 #2 ]', ), ), name='set_body_heat_08')
```

### N40 line 136 (mdb.models)

```text
0133: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0:2 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0:2 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 137 (mdb.models)

```text
0134:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0135:     '[#0:2 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0:2 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 138 (set_body_heat_)

```text
0135:     '[#0:2 #8 ]', ), ), name='set_body_heat_07')
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0:2 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#0 #80000000 ]', ), ), name='set_body_heat_09')
```

### N40 line 139 (mdb.models)

```text
0136: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0:2 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#0 #80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 140 (mdb.models)

```text
0137:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0138:     '[#0:2 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#0 #80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 141 (set_body_heat_)

```text
0138:     '[#0:2 #2 ]', ), ), name='set_body_heat_08')
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#0 #80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#0 #20000000 ]', ), ), name='set_body_heat_10')
```

### N40 line 142 (mdb.models)

```text
0139: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#0 #80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#0 #20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 143 (mdb.models)

```text
0140:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0141:     '[#0 #80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#0 #20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 144 (set_body_heat_)

```text
0141:     '[#0 #80000000 ]', ), ), name='set_body_heat_09')
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#0 #20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#0 #8000000 ]', ), ), name='set_body_heat_11')
```

### N40 line 145 (mdb.models)

```text
0142: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#0 #20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#0 #8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 146 (mdb.models)

```text
0143:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0144:     '[#0 #20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#0 #8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 147 (set_body_heat_)

```text
0144:     '[#0 #20000000 ]', ), ), name='set_body_heat_10')
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#0 #8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#0 #2000000 ]', ), ), name='set_body_heat_12')
```

### N40 line 148 (mdb.models)

```text
0145: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#0 #8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#0 #2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 149 (mdb.models)

```text
0146:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0147:     '[#0 #8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#0 #2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 150 (set_body_heat_)

```text
0147:     '[#0 #8000000 ]', ), ), name='set_body_heat_11')
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#0 #2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#0 #800000 ]', ), ), name='set_body_heat_13')
```

### N40 line 151 (mdb.models)

```text
0148: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#0 #2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#0 #800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 152 (mdb.models)

```text
0149:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0150:     '[#0 #2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#0 #800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 153 (set_body_heat_)

```text
0150:     '[#0 #2000000 ]', ), ), name='set_body_heat_12')
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#0 #800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#0 #200000 ]', ), ), name='set_body_heat_14')
```

### N40 line 154 (mdb.models)

```text
0151: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#0 #800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#0 #200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 155 (mdb.models)

```text
0152:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0153:     '[#0 #800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#0 #200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 156 (set_body_heat_)

```text
0153:     '[#0 #800000 ]', ), ), name='set_body_heat_13')
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#0 #200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#0 #80000 ]', ), ), name='set_body_heat_15')
```

### N40 line 157 (mdb.models)

```text
0154: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#0 #200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#0 #80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 158 (mdb.models)

```text
0155:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0156:     '[#0 #200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#0 #80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 159 (set_body_heat_)

```text
0156:     '[#0 #200000 ]', ), ), name='set_body_heat_14')
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#0 #80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#0 #20000 ]', ), ), name='set_body_heat_16')
```

### N40 line 160 (mdb.models)

```text
0157: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#0 #80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#0 #20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 161 (mdb.models)

```text
0158:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0159:     '[#0 #80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#0 #20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 162 (set_body_heat_)

```text
0159:     '[#0 #80000 ]', ), ), name='set_body_heat_15')
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#0 #20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#0 #8000 ]', ), ), name='set_body_heat_17')
```

### N40 line 163 (mdb.models)

```text
0160: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#0 #20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#0 #8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 164 (mdb.models)

```text
0161:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0162:     '[#0 #20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#0 #8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 165 (set_body_heat_)

```text
0162:     '[#0 #20000 ]', ), ), name='set_body_heat_16')
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#0 #8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#0 #2000 ]', ), ), name='set_body_heat_18')
```

### N40 line 166 (mdb.models)

```text
0163: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#0 #8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#0 #2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 167 (mdb.models)

```text
0164:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0165:     '[#0 #8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#0 #2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 168 (set_body_heat_)

```text
0165:     '[#0 #8000 ]', ), ), name='set_body_heat_17')
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#0 #2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#0 #800 ]', ), ), name='set_body_heat_19')
```

### N40 line 169 (mdb.models)

```text
0166: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#0 #2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#0 #800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 170 (mdb.models)

```text
0167:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0168:     '[#0 #2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#0 #800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 171 (set_body_heat_)

```text
0168:     '[#0 #2000 ]', ), ), name='set_body_heat_18')
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#0 #800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#0 #200 ]', ), ), name='set_body_heat_20')
```

### N40 line 172 (mdb.models)

```text
0169: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#0 #800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#0 #200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 173 (mdb.models)

```text
0170:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0171:     '[#0 #800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#0 #200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 174 (set_body_heat_)

```text
0171:     '[#0 #800 ]', ), ), name='set_body_heat_19')
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#0 #200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#0 #80 ]', ), ), name='set_body_heat_21')
```

### N40 line 175 (mdb.models)

```text
0172: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#0 #200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#0 #80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 176 (mdb.models)

```text
0173:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0174:     '[#0 #200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#0 #80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 177 (set_body_heat_)

```text
0174:     '[#0 #200 ]', ), ), name='set_body_heat_20')
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#0 #80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#0 #20 ]', ), ), name='set_body_heat_22')
```

### N40 line 178 (mdb.models)

```text
0175: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#0 #80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#0 #20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 179 (mdb.models)

```text
0176:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0177:     '[#0 #80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#0 #20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 180 (set_body_heat_)

```text
0177:     '[#0 #80 ]', ), ), name='set_body_heat_21')
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#0 #20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#0 #8 ]', ), ), name='set_body_heat_23')
```

### N40 line 181 (mdb.models)

```text
0178: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#0 #20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#0 #8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 182 (mdb.models)

```text
0179:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0180:     '[#0 #20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#0 #8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0185:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 183 (set_body_heat_)

```text
0180:     '[#0 #20 ]', ), ), name='set_body_heat_22')
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#0 #8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0185:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0186:     '[#0 #2 ]', ), ), name='set_body_heat_24')
```

### N40 line 184 (mdb.models)

```text
0181: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#0 #8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0185:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0186:     '[#0 #2 ]', ), ), name='set_body_heat_24')
0187: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 185 (mdb.models)

```text
0182:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0183:     '[#0 #8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0185:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0186:     '[#0 #2 ]', ), ), name='set_body_heat_24')
0187: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0188:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 186 (set_body_heat_)

```text
0183:     '[#0 #8 ]', ), ), name='set_body_heat_23')
0184: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0185:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0186:     '[#0 #2 ]', ), ), name='set_body_heat_24')
0187: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0188:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0189:     '[#80000000 ]', ), ), name='set_body_heat_25')
```

### N40 line 187 (mdb.models)

```text
0184: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0185:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0186:     '[#0 #2 ]', ), ), name='set_body_heat_24')
0187: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0188:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0189:     '[#80000000 ]', ), ), name='set_body_heat_25')
0190: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 188 (mdb.models)

```text
0185:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0186:     '[#0 #2 ]', ), ), name='set_body_heat_24')
0187: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0188:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0189:     '[#80000000 ]', ), ), name='set_body_heat_25')
0190: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0191:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 189 (set_body_heat_)

```text
0186:     '[#0 #2 ]', ), ), name='set_body_heat_24')
0187: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0188:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0189:     '[#80000000 ]', ), ), name='set_body_heat_25')
0190: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0191:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0192:     '[#20000000 ]', ), ), name='set_body_heat_26')
```

### N40 line 190 (mdb.models)

```text
0187: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0188:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0189:     '[#80000000 ]', ), ), name='set_body_heat_25')
0190: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0191:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0192:     '[#20000000 ]', ), ), name='set_body_heat_26')
0193: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 191 (mdb.models)

```text
0188:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0189:     '[#80000000 ]', ), ), name='set_body_heat_25')
0190: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0191:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0192:     '[#20000000 ]', ), ), name='set_body_heat_26')
0193: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0194:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 192 (set_body_heat_)

```text
0189:     '[#80000000 ]', ), ), name='set_body_heat_25')
0190: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0191:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0192:     '[#20000000 ]', ), ), name='set_body_heat_26')
0193: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0194:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0195:     '[#8000000 ]', ), ), name='set_body_heat_27')
```

### N40 line 193 (mdb.models)

```text
0190: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0191:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0192:     '[#20000000 ]', ), ), name='set_body_heat_26')
0193: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0194:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0195:     '[#8000000 ]', ), ), name='set_body_heat_27')
0196: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 194 (mdb.models)

```text
0191:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0192:     '[#20000000 ]', ), ), name='set_body_heat_26')
0193: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0194:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0195:     '[#8000000 ]', ), ), name='set_body_heat_27')
0196: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0197:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 195 (set_body_heat_)

```text
0192:     '[#20000000 ]', ), ), name='set_body_heat_26')
0193: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0194:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0195:     '[#8000000 ]', ), ), name='set_body_heat_27')
0196: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0197:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0198:     '[#2000000 ]', ), ), name='set_body_heat_28')
```

### N40 line 196 (mdb.models)

```text
0193: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0194:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0195:     '[#8000000 ]', ), ), name='set_body_heat_27')
0196: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0197:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0198:     '[#2000000 ]', ), ), name='set_body_heat_28')
0199: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 197 (mdb.models)

```text
0194:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0195:     '[#8000000 ]', ), ), name='set_body_heat_27')
0196: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0197:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0198:     '[#2000000 ]', ), ), name='set_body_heat_28')
0199: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0200:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 198 (set_body_heat_)

```text
0195:     '[#8000000 ]', ), ), name='set_body_heat_27')
0196: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0197:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0198:     '[#2000000 ]', ), ), name='set_body_heat_28')
0199: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0200:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0201:     '[#800000 ]', ), ), name='set_body_heat_29')
```

### N40 line 199 (mdb.models)

```text
0196: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0197:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0198:     '[#2000000 ]', ), ), name='set_body_heat_28')
0199: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0200:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0201:     '[#800000 ]', ), ), name='set_body_heat_29')
0202: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 200 (mdb.models)

```text
0197:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0198:     '[#2000000 ]', ), ), name='set_body_heat_28')
0199: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0200:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0201:     '[#800000 ]', ), ), name='set_body_heat_29')
0202: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0203:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 201 (set_body_heat_)

```text
0198:     '[#2000000 ]', ), ), name='set_body_heat_28')
0199: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0200:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0201:     '[#800000 ]', ), ), name='set_body_heat_29')
0202: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0203:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0204:     '[#200000 ]', ), ), name='set_body_heat_30')
```

### N40 line 202 (mdb.models)

```text
0199: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0200:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0201:     '[#800000 ]', ), ), name='set_body_heat_29')
0202: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0203:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0204:     '[#200000 ]', ), ), name='set_body_heat_30')
0205: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 203 (mdb.models)

```text
0200:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0201:     '[#800000 ]', ), ), name='set_body_heat_29')
0202: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0203:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0204:     '[#200000 ]', ), ), name='set_body_heat_30')
0205: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0206:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 204 (set_body_heat_)

```text
0201:     '[#800000 ]', ), ), name='set_body_heat_29')
0202: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0203:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0204:     '[#200000 ]', ), ), name='set_body_heat_30')
0205: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0206:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0207:     '[#80000 ]', ), ), name='set_body_heat_31')
```

### N40 line 205 (mdb.models)

```text
0202: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0203:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0204:     '[#200000 ]', ), ), name='set_body_heat_30')
0205: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0206:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0207:     '[#80000 ]', ), ), name='set_body_heat_31')
0208: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 206 (mdb.models)

```text
0203:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0204:     '[#200000 ]', ), ), name='set_body_heat_30')
0205: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0206:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0207:     '[#80000 ]', ), ), name='set_body_heat_31')
0208: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0209:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 207 (set_body_heat_)

```text
0204:     '[#200000 ]', ), ), name='set_body_heat_30')
0205: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0206:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0207:     '[#80000 ]', ), ), name='set_body_heat_31')
0208: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0209:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0210:     '[#20000 ]', ), ), name='set_body_heat_32')
```

### N40 line 208 (mdb.models)

```text
0205: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0206:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0207:     '[#80000 ]', ), ), name='set_body_heat_31')
0208: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0209:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0210:     '[#20000 ]', ), ), name='set_body_heat_32')
0211: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 209 (mdb.models)

```text
0206:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0207:     '[#80000 ]', ), ), name='set_body_heat_31')
0208: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0209:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0210:     '[#20000 ]', ), ), name='set_body_heat_32')
0211: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0212:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 210 (set_body_heat_)

```text
0207:     '[#80000 ]', ), ), name='set_body_heat_31')
0208: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0209:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0210:     '[#20000 ]', ), ), name='set_body_heat_32')
0211: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0212:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0213:     '[#8000 ]', ), ), name='set_body_heat_33')
```

### N40 line 211 (mdb.models)

```text
0208: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0209:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0210:     '[#20000 ]', ), ), name='set_body_heat_32')
0211: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0212:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0213:     '[#8000 ]', ), ), name='set_body_heat_33')
0214: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 212 (mdb.models)

```text
0209:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0210:     '[#20000 ]', ), ), name='set_body_heat_32')
0211: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0212:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0213:     '[#8000 ]', ), ), name='set_body_heat_33')
0214: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0215:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 213 (set_body_heat_)

```text
0210:     '[#20000 ]', ), ), name='set_body_heat_32')
0211: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0212:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0213:     '[#8000 ]', ), ), name='set_body_heat_33')
0214: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0215:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0216:     '[#2000 ]', ), ), name='set_body_heat_34')
```

### N40 line 214 (mdb.models)

```text
0211: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0212:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0213:     '[#8000 ]', ), ), name='set_body_heat_33')
0214: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0215:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0216:     '[#2000 ]', ), ), name='set_body_heat_34')
0217: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 215 (mdb.models)

```text
0212:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0213:     '[#8000 ]', ), ), name='set_body_heat_33')
0214: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0215:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0216:     '[#2000 ]', ), ), name='set_body_heat_34')
0217: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0218:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 216 (set_body_heat_)

```text
0213:     '[#8000 ]', ), ), name='set_body_heat_33')
0214: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0215:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0216:     '[#2000 ]', ), ), name='set_body_heat_34')
0217: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0218:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0219:     '[#800 ]', ), ), name='set_body_heat_35')
```

### N40 line 217 (mdb.models)

```text
0214: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0215:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0216:     '[#2000 ]', ), ), name='set_body_heat_34')
0217: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0218:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0219:     '[#800 ]', ), ), name='set_body_heat_35')
0220: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 218 (mdb.models)

```text
0215:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0216:     '[#2000 ]', ), ), name='set_body_heat_34')
0217: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0218:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0219:     '[#800 ]', ), ), name='set_body_heat_35')
0220: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0221:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 219 (set_body_heat_)

```text
0216:     '[#2000 ]', ), ), name='set_body_heat_34')
0217: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0218:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0219:     '[#800 ]', ), ), name='set_body_heat_35')
0220: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0221:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0222:     '[#200 ]', ), ), name='set_body_heat_36')
```

### N40 line 220 (mdb.models)

```text
0217: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0218:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0219:     '[#800 ]', ), ), name='set_body_heat_35')
0220: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0221:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0222:     '[#200 ]', ), ), name='set_body_heat_36')
0223: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 221 (mdb.models)

```text
0218:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0219:     '[#800 ]', ), ), name='set_body_heat_35')
0220: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0221:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0222:     '[#200 ]', ), ), name='set_body_heat_36')
0223: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0224:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 222 (set_body_heat_)

```text
0219:     '[#800 ]', ), ), name='set_body_heat_35')
0220: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0221:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0222:     '[#200 ]', ), ), name='set_body_heat_36')
0223: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0224:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0225:     '[#80 ]', ), ), name='set_body_heat_37')
```

### N40 line 223 (mdb.models)

```text
0220: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0221:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0222:     '[#200 ]', ), ), name='set_body_heat_36')
0223: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0224:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0225:     '[#80 ]', ), ), name='set_body_heat_37')
0226: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 224 (mdb.models)

```text
0221:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0222:     '[#200 ]', ), ), name='set_body_heat_36')
0223: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0224:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0225:     '[#80 ]', ), ), name='set_body_heat_37')
0226: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0227:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 225 (set_body_heat_)

```text
0222:     '[#200 ]', ), ), name='set_body_heat_36')
0223: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0224:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0225:     '[#80 ]', ), ), name='set_body_heat_37')
0226: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0227:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0228:     '[#20 ]', ), ), name='set_body_heat_38')
```

### N40 line 226 (mdb.models)

```text
0223: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0224:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0225:     '[#80 ]', ), ), name='set_body_heat_37')
0226: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0227:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0228:     '[#20 ]', ), ), name='set_body_heat_38')
0229: mdb.models['Model-1'].parts['part_plate'].Set(faces=
```

### N40 line 227 (mdb.models)

```text
0224:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0225:     '[#80 ]', ), ), name='set_body_heat_37')
0226: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0227:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0228:     '[#20 ]', ), ), name='set_body_heat_38')
0229: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0230:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 228 (set_body_heat_)

```text
0225:     '[#80 ]', ), ), name='set_body_heat_37')
0226: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0227:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0228:     '[#20 ]', ), ), name='set_body_heat_38')
0229: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0230:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0231:     '[#8 ]', ), ), name='set_body_heat_39')
```

### N40 line 229 (mdb.models)

```text
0226: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0227:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0228:     '[#20 ]', ), ), name='set_body_heat_38')
0229: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0230:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0231:     '[#8 ]', ), ), name='set_body_heat_39')
0232: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
```

### N40 line 230 (mdb.models)

```text
0227:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0228:     '[#20 ]', ), ), name='set_body_heat_38')
0229: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0230:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0231:     '[#8 ]', ), ), name='set_body_heat_39')
0232: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0233:     side1Edges=
```

### N40 line 231 (set_body_heat_)

```text
0228:     '[#20 ]', ), ), name='set_body_heat_38')
0229: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0230:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0231:     '[#8 ]', ), ), name='set_body_heat_39')
0232: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0233:     side1Edges=
0234:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
```

### N40 line 232 (mdb.models)

```text
0229: mdb.models['Model-1'].parts['part_plate'].Set(faces=
0230:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0231:     '[#8 ]', ), ), name='set_body_heat_39')
0232: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0233:     side1Edges=
0234:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0235:     '[#94a54966 #a5294a52 #294a5294 #4a5294a5 #5294a529 #94a5294a #eca52 ]', ),
```

### N40 line 234 (mdb.models)

```text
0231:     '[#8 ]', ), ), name='set_body_heat_39')
0232: mdb.models['Model-1'].parts['part_plate'].Surface(name='surf_external_all',
0233:     side1Edges=
0234:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0235:     '[#94a54966 #a5294a52 #294a5294 #4a5294a5 #5294a529 #94a5294a #eca52 ]', ),
0236:     ))
0237: mdb.models['Model-1'].Material(name='SS316L For AM')
```

### N40 line 237 (mdb.models)

```text
0234:     mdb.models['Model-1'].parts['part_plate'].edges.getSequenceFromMask((
0235:     '[#94a54966 #a5294a52 #294a5294 #4a5294a5 #5294a529 #94a5294a #eca52 ]', ),
0236:     ))
0237: mdb.models['Model-1'].Material(name='SS316L For AM')
0238: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0239:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0240:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
```

### N40 line 238 (mdb.models)

```text
0235:     '[#94a54966 #a5294a52 #294a5294 #4a5294a5 #5294a529 #94a5294a #eca52 ]', ),
0236:     ))
0237: mdb.models['Model-1'].Material(name='SS316L For AM')
0238: mdb.models['Model-1'].materials['SS316L For AM'].Conductivity(dependencies=0,
0239:     table=((14.0, 20.0), (16.0, 100.0), (17.0, 200.0), (19.0, 400.0), (21.5,
0240:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0241:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
```

### N40 line 243 (mdb.models)

```text
0240:     600.0), (24.0, 800.0), (26.5, 1000.0), (29.0, 1200.0), (31.0, 1375.0), (
0241:     29.0, 1400.0), (29.0, 1723.0), (29.0, 3000.0)), temperatureDependency=ON,
0242:     type=ISOTROPIC)
0243: mdb.models['Model-1'].materials['SS316L For AM'].Density(dependencies=0,
0244:     distributionType=UNIFORM, fieldName='', table=((7980.0, 20.0), (7950.0,
0245:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0246:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
```

### N40 line 248 (mdb.models)

```text
0245:     100.0), (7920.0, 200.0), (7860.0, 400.0), (7800.0, 600.0), (7740.0, 800.0),
0246:     (7680.0, 1000.0), (7620.0, 1200.0), (7580.0, 1375.0), (7450.0, 1400.0), (
0247:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0248: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0249:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0250: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0251:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
```

### N40 line 250 (mdb.models)

```text
0247:     7300.0, 1723.0), (7200.0, 3000.0)), temperatureDependency=ON)
0248: mdb.models['Model-1'].materials['SS316L For AM'].setValues(description=
0249:     'Material property of AISI Type 316L Steel in Additive Manufacturing\n')
0250: mdb.models['Model-1'].materials['SS316L For AM'].Elastic(dependencies=0,
0251:     moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((208000000000.0,
0252:     0.3, 20.0), (202000000000.0, 0.3, 100.0), (194000000000.0, 0.3, 200.0), (
0253:     178000000000.0, 0.3, 400.0), (162000000000.0, 0.3, 600.0), (142000000000.0,
```

### N40 line 258 (mdb.models)

```text
0255:     15000000000.0, 0.3, 1375.0), (100000000.0, 0.3, 1400.0), (10000000.0, 0.3,
0256:     1723.0), (1000000.0, 0.3, 3000.0)), temperatureDependency=ON, type=
0257:     ISOTROPIC)
0258: mdb.models['Model-1'].materials['SS316L For AM'].Expansion(dependencies=0,
0259:     table=((1.48e-05, 20.0), (1.6e-05, 100.0), (1.68e-05, 200.0), (1.78e-05,
0260:     400.0), (1.87e-05, 600.0), (1.96e-05, 800.0), (2.02e-05, 1000.0), (
0261:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
```

### N40 line 264 (mdb.models)

```text
0261:     2.08e-05, 1200.0), (2.15e-05, 1375.0), (2.2e-05, 1400.0), (2.2e-05,
0262:     1723.0), (2.2e-05, 3000.0)), temperatureDependency=ON, type=ISOTROPIC,
0263:     userSubroutine=OFF, zero=0.0)
0264: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0265:     1375.0, 1400.0), ))
0266: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0267:     '')
```

### N40 line 266 (mdb.models)

```text
0263:     userSubroutine=OFF, zero=0.0)
0264: mdb.models['Model-1'].materials['SS316L For AM'].LatentHeat(table=((256000.0,
0265:     1375.0, 1400.0), ))
0266: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0267:     '')
0268: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0269:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
```

### N40 line 268 (mdb.models)

```text
0265:     1375.0, 1400.0), ))
0266: mdb.models['Model-1'].materials['SS316L For AM'].setValues(materialIdentifier=
0267:     '')
0268: mdb.models['Model-1'].materials['SS316L For AM'].Plastic(dataType=HALF_CYCLE,
0269:     dependencies=0, extrapolation=CONSTANT, hardening=ISOTROPIC,
0270:     numBackstresses=1, rate=OFF, scaleStress=None, staticRecovery=OFF,
0271:     strainRangeDependency=OFF, table=((580000000.0, 0.0, 20.0), (530000000.0,
```

### N40 line 277 (mdb.models)

```text
0274:     1000.0), (30000000.0, 0.0, 1200.0), (2000000.0, 0.0, 1375.0), (10000.0,
0275:     0.0, 1400.0), (5000.0, 0.0, 1723.0), (1000.0, 0.0, 3000.0)),
0276:     temperatureDependency=ON)
0277: mdb.models['Model-1'].materials['SS316L For AM'].SpecificHeat(dependencies=0,
0278:     law=CONSTANTVOLUME, table=((450.0, 20.0), (480.0, 100.0), (505.0, 200.0), (
0279:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0280:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
```

### N40 line 282 (mdb.models)

```text
0279:     540.0, 400.0), (570.0, 600.0), (600.0, 800.0), (635.0, 1000.0), (670.0,
0280:     1200.0), (700.0, 1375.0), (750.0, 1400.0), (760.0, 1723.0), (800.0,
0281:     3000.0)), temperatureDependency=ON)
0282: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0283:     'property_section_all', thickness=None)
0284: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0285:     offsetField='', offsetType=MIDDLE_SURFACE, region=
```

### N40 line 284 (mdb.models)

```text
0281:     3000.0)), temperatureDependency=ON)
0282: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0283:     'property_section_all', thickness=None)
0284: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0285:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0286:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0287:     'property_section_all', thicknessAssignment=FROM_SECTION)
```

### N40 line 285 (region=)

```text
0282: mdb.models['Model-1'].HomogeneousSolidSection(material='SS316L For AM', name=
0283:     'property_section_all', thickness=None)
0284: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0285:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0286:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0287:     'property_section_all', thicknessAssignment=FROM_SECTION)
0288: mdb.models['Model-1'].setValues(absoluteZero=-173.15, stefanBoltzmann=5.67e-08)
```

### N40 line 286 (mdb.models)

```text
0283:     'property_section_all', thickness=None)
0284: mdb.models['Model-1'].parts['part_plate'].SectionAssignment(offset=0.0,
0285:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0286:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0287:     'property_section_all', thicknessAssignment=FROM_SECTION)
0288: mdb.models['Model-1'].setValues(absoluteZero=-173.15, stefanBoltzmann=5.67e-08)
0289: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
```

### N40 line 288 (mdb.models)

```text
0285:     offsetField='', offsetType=MIDDLE_SURFACE, region=
0286:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0287:     'property_section_all', thicknessAssignment=FROM_SECTION)
0288: mdb.models['Model-1'].setValues(absoluteZero=-173.15, stefanBoltzmann=5.67e-08)
0289: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0290: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0291:     part=mdb.models['Model-1'].parts['part_plate'])
```

### N40 line 289 (mdb.models)

```text
0286:     mdb.models['Model-1'].parts['part_plate'].sets['section_all'], sectionName=
0287:     'property_section_all', thicknessAssignment=FROM_SECTION)
0288: mdb.models['Model-1'].setValues(absoluteZero=-173.15, stefanBoltzmann=5.67e-08)
0289: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0290: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0291:     part=mdb.models['Model-1'].parts['part_plate'])
0292: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
```

### N40 line 290 (mdb.models)

```text
0287:     'property_section_all', thicknessAssignment=FROM_SECTION)
0288: mdb.models['Model-1'].setValues(absoluteZero=-173.15, stefanBoltzmann=5.67e-08)
0289: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0290: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0291:     part=mdb.models['Model-1'].parts['part_plate'])
0292: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0293:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
```

### N40 line 291 (mdb.models)

```text
0288: mdb.models['Model-1'].setValues(absoluteZero=-173.15, stefanBoltzmann=5.67e-08)
0289: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0290: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0291:     part=mdb.models['Model-1'].parts['part_plate'])
0292: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0293:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0294:     nlgeom=ON, previous='Initial', timePeriod=0.2)
```

### N40 line 292 (mdb.models;CoupledTempDisplacementStep)

```text
0289: mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
0290: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0291:     part=mdb.models['Model-1'].parts['part_plate'])
0292: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0293:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0294:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0295: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.1,
```

### N40 line 293 (step_scan_)

```text
0290: mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='inst_plate',
0291:     part=mdb.models['Model-1'].parts['part_plate'])
0292: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0293:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0294:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0295: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.1,
0296:     maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
```

### N40 line 295 (mdb.models;CoupledTempDisplacementStep)

```text
0292: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=
0293:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0294:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0295: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.1,
0296:     maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0297:     previous='step_scan_00', timePeriod=3.4)
0298: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
```

### N40 line 296 (step_cool_)

```text
0293:     0.001, maxInc=0.01, maxNumInc=999999, minInc=2e-30, name='step_scan_00',
0294:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0295: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.1,
0296:     maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0297:     previous='step_scan_00', timePeriod=3.4)
0298: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0299:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
```

### N40 line 297 (step_scan_)

```text
0294:     nlgeom=ON, previous='Initial', timePeriod=0.2)
0295: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.1,
0296:     maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0297:     previous='step_scan_00', timePeriod=3.4)
0298: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0299:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0300: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
```

### N40 line 298 (mdb.models)

```text
0295: mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=400.0, initialInc=0.1,
0296:     maxInc=0.2, maxNumInc=999999, minInc=3.4e-30, name='step_cool_00',
0297:     previous='step_scan_00', timePeriod=3.4)
0298: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0299:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0300: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0301:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
```

### N40 line 300 (mdb.models)

```text
0297:     previous='step_scan_00', timePeriod=3.4)
0298: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
0299:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0300: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0301:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0302: mdb.models['Model-1'].FilmCondition(createStepName='step_scan_00', definition=
0303:     EMBEDDED_COEFF, filmCoeff=46.5, filmCoeffAmplitude='', name=
```

### N40 line 302 (step_scan_;mdb.models;createStepName)

```text
0299:     'NT', 'S', 'U', 'PEEQ', 'RF', 'HFL'))
0300: mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
0301:     'ALLIE', 'ALLKE', 'ALLSE', 'ALLPD', 'ALLAE', 'ALLWK', 'ETOTAL', 'ALLSD'))
0302: mdb.models['Model-1'].FilmCondition(createStepName='step_scan_00', definition=
0303:     EMBEDDED_COEFF, filmCoeff=46.5, filmCoeffAmplitude='', name=
0304:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0305:     sinkFieldName='', sinkTemperature=20.0, surface=
```

### N40 line 306 (mdb.models)

```text
0303:     EMBEDDED_COEFF, filmCoeff=46.5, filmCoeffAmplitude='', name=
0304:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0305:     sinkFieldName='', sinkTemperature=20.0, surface=
0306:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0307: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0308:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0309:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
```

### N40 line 307 (mdb.models)

```text
0304:     'film_external_cooling', sinkAmplitude='', sinkDistributionType=UNIFORM,
0305:     sinkFieldName='', sinkTemperature=20.0, surface=
0306:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0307: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0308:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0309:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0310:     radiationType=AMBIENT, surface=
```

### N40 line 308 (step_scan_;createStepName)

```text
0305:     sinkFieldName='', sinkTemperature=20.0, surface=
0306:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0307: mdb.models['Model-1'].RadiationToAmbient(ambientTemperature=20.0,
0308:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0309:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0310:     radiationType=AMBIENT, surface=
0311:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
```

### N40 line 311 (mdb.models)

```text
0308:     ambientTemperatureAmp='', createStepName='step_scan_00', distributionType=
0309:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0310:     radiationType=AMBIENT, surface=
0311:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0312: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0313:     80000000000.0, name='load_body_hflux_00', region=
0314:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
```

### N40 line 312 (step_scan_;BodyHeatFlux;mdb.models;createStepName)

```text
0309:     UNIFORM, emissivity=0.285, field='', name='rad_external_ambient',
0310:     radiationType=AMBIENT, surface=
0311:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0312: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0313:     80000000000.0, name='load_body_hflux_00', region=
0314:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0315: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
```

### N40 line 313 (load_body_hflux_;region=)

```text
0310:     radiationType=AMBIENT, surface=
0311:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0312: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0313:     80000000000.0, name='load_body_hflux_00', region=
0314:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0315: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0316: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N40 line 314 (set_body_heat_;mdb.models)

```text
0311:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].surfaces['surf_external_all'])
0312: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0313:     80000000000.0, name='load_body_hflux_00', region=
0314:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0315: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0316: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0317:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
```

### N40 line 315 (step_cool_;load_body_hflux_;mdb.models;loads[;deactivate)

```text
0312: mdb.models['Model-1'].BodyHeatFlux(createStepName='step_scan_00', magnitude=
0313:     80000000000.0, name='load_body_hflux_00', region=
0314:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0315: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0316: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0317:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0318:     region=Region(
```

### N40 line 316 (mdb.models;createStepName)

```text
0313:     80000000000.0, name='load_body_hflux_00', region=
0314:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['set_body_heat_00'])
0315: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0316: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0317:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0318:     region=Region(
0319:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N40 line 318 (region=;Region)

```text
0315: mdb.models['Model-1'].loads['load_body_hflux_00'].deactivate('step_cool_00')
0316: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0317:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0318:     region=Region(
0319:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0320:     mask=('[#0:3 #40000000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0321: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
```

### N40 line 319 (mdb.models)

```text
0316: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0317:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_xy',
0318:     region=Region(
0319:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0320:     mask=('[#0:3 #40000000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0321: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0322:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
```

### N40 line 321 (mdb.models;createStepName)

```text
0318:     region=Region(
0319:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0320:     mask=('[#0:3 #40000000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0321: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0322:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0323:     region=Region(
0324:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
```

### N40 line 323 (region=;Region)

```text
0320:     mask=('[#0:3 #40000000 ]', ), )), u1=SET, u2=SET, ur3=UNSET)
0321: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0322:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0323:     region=Region(
0324:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0325:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0326: mdb.models['Model-1'].Temperature(createStepName='Initial',
```

### N40 line 324 (mdb.models)

```text
0321: mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial',
0322:     distributionType=UNIFORM, fieldName='', localCsys=None, name='bc_point_y',
0323:     region=Region(
0324:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0325:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0326: mdb.models['Model-1'].Temperature(createStepName='Initial',
0327:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
```

### N40 line 326 (mdb.models;createStepName)

```text
0323:     region=Region(
0324:     vertices=mdb.models['Model-1'].rootAssembly.instances['inst_plate'].vertices.getSequenceFromMask(
0325:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0326: mdb.models['Model-1'].Temperature(createStepName='Initial',
0327:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0328:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0329:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
```

### N40 line 328 (region=)

```text
0325:     mask=('[#4 ]', ), )), u1=UNSET, u2=SET, ur3=UNSET)
0326: mdb.models['Model-1'].Temperature(createStepName='Initial',
0327:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0328:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0329:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0330: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0331:     minSizeFactor=0.1, size=0.0005)
```

### N40 line 329 (mdb.models)

```text
0326: mdb.models['Model-1'].Temperature(createStepName='Initial',
0327:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0328:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0329:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0330: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0331:     minSizeFactor=0.1, size=0.0005)
0332: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
```

### N40 line 330 (mdb.models)

```text
0327:     crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=
0328:     UNIFORM, magnitudes=(20.0, ), name='predefined_temperature_all', region=
0329:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0330: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0331:     minSizeFactor=0.1, size=0.0005)
0332: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0333:     regions=
```

### N40 line 332 (mdb.models)

```text
0329:     mdb.models['Model-1'].rootAssembly.instances['inst_plate'].sets['section_all'])
0330: mdb.models['Model-1'].parts['part_plate'].seedPart(deviationFactor=0.1,
0331:     minSizeFactor=0.1, size=0.0005)
0332: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0333:     regions=
0334:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0335:     '[#ffffffff:2 #fffff ]', ), ), technique=STRUCTURED)
```

### N40 line 334 (mdb.models)

```text
0331:     minSizeFactor=0.1, size=0.0005)
0332: mdb.models['Model-1'].parts['part_plate'].setMeshControls(elemShape=QUAD,
0333:     regions=
0334:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0335:     '[#ffffffff:2 #fffff ]', ), ), technique=STRUCTURED)
0336: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0337:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
```

### N40 line 336 (mdb.models)

```text
0333:     regions=
0334:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0335:     '[#ffffffff:2 #fffff ]', ), ), technique=STRUCTURED)
0336: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0337:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0338:     elemLibrary=STANDARD)), regions=(
0339:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
```

### N40 line 339 (mdb.models)

```text
0336: mdb.models['Model-1'].parts['part_plate'].setElementType(elemTypes=(ElemType(
0337:     elemCode=CPE4T, elemLibrary=STANDARD), ElemType(elemCode=CPE3T,
0338:     elemLibrary=STANDARD)), regions=(
0339:     mdb.models['Model-1'].parts['part_plate'].faces.getSequenceFromMask((
0340:     '[#ffffffff:2 #fffff ]', ), ), ))
0341: # Save by wuxia on 2026_06_10-22.58.49; build 2024 2023_09_21-20.55.25 RELr426 190762
0342: from part import *
```

### N40 line 355 (mdb.models)

```text
0352: from sketch import *
0353: from visualization import *
0354: from connectorBehavior import *
0355: mdb.models['Model-1'].rootAssembly.regenerate()
0356: # Save by wuxia on 2026_06_10-22.59.16; build 2024 2023_09_21-20.55.25 RELr426 190762
```
