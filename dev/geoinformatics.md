# Geoinformatics

Includes Geodesy/Cartography/GIS

- Universal Traverse Mercator (UTM)
    - http://www.jaworski.ca/utmzones.htm
- Earth Centered Earth Fixed (ECEF)
- Geodetic Reference System 1980 (GRS80)
- World Geodetic System 1984 (WGS84)
- EPSG:5186
    - used in South Korea
    - based on GRS80 ellipsoid
- EPSG:4326
    - Uses WGS84 Lat/Lon
    - the default CRS used by the Global Positioning System (GPS)
- EPSG:3857
    - Web Mercator
    - Used in Google Maps, OpenStreetMap, and Bing Maps.
- EPSG:4978

## SRTM GL1 v3.0

- comprises 14297 `.tif` tile files
- 3601x3601 or 1801x3601
- each contains gdal header
    - see the example below

### The image header of n53_w086_1arc_v3.tif

|Name|Value|
|---|---|
|ExifTool Version Number[![](https://exif.tools/external.svg)](https://exif.tools/meta/ExifTool-Version-Number)|12.25|
|File Name[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Name)|phpx4Y3DS|
|Directory[![](https://exif.tools/external.svg)](https://exif.tools/meta/Directory)|/tmp|
|File Size[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Size)|12 MiB|
|File Modification Date/Time[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Modification-Date/Time)|2025:08:31 23:19:44+00:00|
|File Access Date/Time[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Access-Date/Time)|2025:08:31 23:19:37+00:00|
|File Inode Change Date/Time[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Inode-Change-Date/Time)|2025:08:31 23:19:44+00:00|
|File Permissions[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Permissions)|-rw-------|
|File Type[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Type)|TIFF|
|File Type Extension[![](https://exif.tools/external.svg)](https://exif.tools/meta/File-Type-Extension)|tif|
|MIME Type[![](https://exif.tools/external.svg)](https://exif.tools/meta/MIME-Type)|image/tiff|
|Exif Byte Order[![](https://exif.tools/external.svg)](https://exif.tools/meta/Exif-Byte-Order)|Little-endian (Intel, II)|
|Image Width[![](https://exif.tools/external.svg)](https://exif.tools/meta/Image-Width)|1801|
|Image Height[![](https://exif.tools/external.svg)](https://exif.tools/meta/Image-Height)|3601|
|Bits Per Sample[![](https://exif.tools/external.svg)](https://exif.tools/meta/Bits-Per-Sample)|16|
|Compression[![](https://exif.tools/external.svg)](https://exif.tools/meta/Compression)|Uncompressed|
|Photometric Interpretation[![](https://exif.tools/external.svg)](https://exif.tools/meta/Photometric-Interpretation)|BlackIsZero|
|Strip Offsets[![](https://exif.tools/external.svg)](https://exif.tools/meta/Strip-Offsets)|(Binary data 14673 bytes, use -b option to extract)|
|Samples Per Pixel[![](https://exif.tools/external.svg)](https://exif.tools/meta/Samples-Per-Pixel)|1|
|Rows Per Strip[![](https://exif.tools/external.svg)](https://exif.tools/meta/Rows-Per-Strip)|2|
|Strip Byte Counts[![](https://exif.tools/external.svg)](https://exif.tools/meta/Strip-Byte-Counts)|(Binary data 9004 bytes, use -b option to extract)|
|Planar Configuration[![](https://exif.tools/external.svg)](https://exif.tools/meta/Planar-Configuration)|Chunky|
|Sample Format[![](https://exif.tools/external.svg)](https://exif.tools/meta/Sample-Format)|Signed|
|Pixel Scale[![](https://exif.tools/external.svg)](https://exif.tools/meta/Pixel-Scale)|0.000555555555555556 0.000277777777777778 0|
|Model Tie Point[![](https://exif.tools/external.svg)](https://exif.tools/meta/Model-Tie-Point)|0 0 0 -86 54 0|
|GDAL Metadata[![](https://exif.tools/external.svg)](https://exif.tools/meta/GDAL-Metadata)|. 0002. 01. SRTM . 0013. WGS84. 0000. 0000. 0000. A. DTED2. 0530000N. 0860000W. USCNIMA . NA . 0002. U. U . E17 095 . E17 095 . 0004. 0004. E96. m..|
|GDAL No Data[![](https://exif.tools/external.svg)](https://exif.tools/meta/GDAL-No-Data)|-32767|
|Geo Tiff Version[![](https://exif.tools/external.svg)](https://exif.tools/meta/Geo-Tiff-Version)|1.1.0|
|GT Model Type[![](https://exif.tools/external.svg)](https://exif.tools/meta/GT-Model-Type)|Geographic|
|GT Raster Type[![](https://exif.tools/external.svg)](https://exif.tools/meta/GT-Raster-Type)|Pixel Is Point|
|Geographic Type[![](https://exif.tools/external.svg)](https://exif.tools/meta/Geographic-Type)|WGS 84|
|Geog Citation[![](https://exif.tools/external.svg)](https://exif.tools/meta/Geog-Citation)|WGS 84|
|Geog Angular Units[![](https://exif.tools/external.svg)](https://exif.tools/meta/Geog-Angular-Units)|Angular Degree|
|Geog Semi Major Axis[![](https://exif.tools/external.svg)](https://exif.tools/meta/Geog-Semi-Major-Axis)|6378137|
|Geog Inv Flattening[![](https://exif.tools/external.svg)](https://exif.tools/meta/Geog-Inv-Flattening)|298.257223563|
|Image Size[![](https://exif.tools/external.svg)](https://exif.tools/meta/Image-Size)|1801x3601|
|Megapixels[![](https://exif.tools/external.svg)](https://exif.tools/meta/Megapixels)|6.5|
