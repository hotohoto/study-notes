# motion-capture

## TODO

- check retargeting papers
    - https://github.com/eth-siplab/AvatarPoser
    - https://siplab.org/projects/EgoPoser
- check SMPL papers
    - https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
    - SMPL, SMPL-X, SMPLer, SMPLest, STAR
        - do they have the same skeleton??
- check license free papers
    - https://github.com/jeffffffli/HybrIK
        - HybrIK, HybrIK-X
        - https://arxiv.org/abs/2304.05690
- check video multi-person motion capture papers

## Glossary

- Human Mesh Recovery (HMR)
    - image to SMPL
- 

## File formats

### Biovision Hierarchy (BVH)

    - https://en.wikipedia.org/wiki/Biovision_Hierarchy
    - https://www.okino.com/conv/imp_bvh.htm
    - text format that contains only the motion data

```bvh
HIERARCHY
ROOT Hips
{
	OFFSET	0.00	0.00	0.00
	CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
	JOINT LeftHip
	{
		OFFSET	 3.29	 0.00	 0.00
		CHANNELS 3 Zrotation Xrotation Yrotation
		JOINT LeftKnee
		{
			OFFSET	 0.00	-16.57	 0.00
			CHANNELS 3 Zrotation Xrotation Yrotation
			JOINT LeftAnkle
			{
				OFFSET	 0.00	-16.55	 0.00
				CHANNELS 3 Zrotation Xrotation Yrotation
				End Site 
				{
					OFFSET	 0.00	-3.30	 0.00
				}
			}
		}
	}
	JOINT RightHip
	{
		OFFSET	-3.29	 0.00	 0.00
		CHANNELS 3 Zrotation Xrotation Yrotation
		JOINT RightKnee
		{
			OFFSET	 0.00	-16.51	 0.00
			CHANNELS 3 Zrotation Xrotation Yrotation
			JOINT RightAnkle
			{
				OFFSET	 0.00	-16.69	 0.00
				CHANNELS 3 Zrotation Xrotation Yrotation
				End Site 
				{
					OFFSET	 0.00	-3.48	 0.00
				}
			}
		}
	}
}
MOTION
Frames:     30
Frame Time: 0.033333
 10.87	 36.65	 13.54	 8.70	 1.67	 91.18	 
10.89	 9.41	-1.80	-2.30	 7.50	 12.08	0.00
-10.56	 0.00	-13.95	 10.19	 2.52	 1.45	 
7.58	-10.90	 0.00	-12.38	 0.00	-2.30	 7.22
-0.10	-0.13	 0.00	 5.48	 48.69	-5.16	 
13.44	-10.40	-22.07	-14.50	-0.21	-5.66	-0.02
7.72	 0.00	-7.17	-52.62	 2.63	-1.95	
-1.90	-20.04	-18.84	 0.16	-4.85	 0.01	 0.00
35.36	 0.00	 1.77	-36.47	-2.39
```

### The other file formats

- Filmbox (FBX)
    - supports multi track
- glTF
- [OpenUSD](open-usd.md)

## Skeletons and datasets

- SMPL
    - the dataset is for non commercial use only
    - parameters
        - $\beta$
            - shape 10D
        - $\theta$
            - pose 72D
            - 24 main joints
                - 0: Pelvis (Root)
                - 1: Left Hip
                - 2: Right Hip
                - 3: Spine1
                - 4: Left Knee
                - 5: Right Knee
                - 6: Spine2
                - 7: Left Ankle
                - 8: Right Ankle
                - 9: Spine3
                - 10: Left Foot
                - 11: Right Foot
                - 12: Neck
                - 13: Left Collar
                - 14: Right Collar
                - 15: Head
                - 16: Left Shoulder
                - 17: Right Shoulder
                - 18: Left Elbow
                - 19: Right Elbow
                - 20: Left Wrist
                - 21: Right Wrist
                - 22: Left Hand
                - 23: Right Hand
        - $T$
            - root translation 3D
    - hips -> spine -> neck -> head
    - shoulder -> elbow -> wrist
    - hip -> knee -> ankle
    - https://github.com/EricGuo5513/HumanML3D
- Mixamo / Adobe HumanIK
    - up to 60 + joints
- OpenPose / COCO human pose
    - https://github.com/robertklee/COCO-Human-Pose/blob/main/README.md
    - 17~25 joints
- Unity Humanoid Rig
- Human3.6M
- MPI-INF-3DHP
- ...

## Retargeting

- direct mapping
    - COCO -> SMPL
    - Mixamo -> SMPL

## Papers

- 

## BVH Viewers

- Blender
- https://lo-th.github.io/olympe/BVH_player.html
- BeeVeeH

## Related pages

- [[pose-estimation]]
