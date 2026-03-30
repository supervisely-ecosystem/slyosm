Restoring Clean, Accurate Polylines from Instance Segmentation Masks: Algorithms, Best Practices, and Practical Pipelines for Road Network Extraction

Introduction

The extraction of road networks from imagery is a foundational task in geospatial analysis, urban planning, autonomous navigation, and digital mapping. With the advent of deep learning, instance segmentation models—such as Mask R-CNN, U-Net, and their derivatives—have become the de facto standard for delineating road surfaces in high-resolution aerial and satellite images. However, the raw output of these models is typically a binary or multi-class mask, which, while effective for pixel-level classification, is not directly suitable for downstream applications that require vectorized, topologically correct, and geometrically accurate polylines. These applications include map generation, routing, GIS analysis, and integration with platforms like OpenStreetMap (OSM) or routing engines such as OSRM and GraphHopper.

The challenge, therefore, is to convert these raster masks into clean, accurate polylines that preserve the geometric and topological structure of the underlying road network. This process—commonly referred to as mask-to-polyline conversion—encompasses a series of algorithmic steps: preprocessing, skeletonization, vectorization, graph construction, and post-processing. Each step must address practical issues such as noise, occlusions, fragmented segments, and the preservation of connectivity and intersection structure.

This report provides a comprehensive, in-depth exploration of the state-of-the-art methods, algorithms, and tools for restoring polylines from segmentation masks. It synthesizes best practices from academic literature, open-source software, and recent research, and offers practical guidance for robust, scalable implementation.

Problem Overview: Goals and Success Criteria for Mask-to-Polyline Conversion

The primary goal of mask-to-polyline conversion is to transform the pixel-wise output of an instance segmentation model into a set of vector polylines that accurately represent the road network. The success of this process is measured by several criteria:

Geometric Accuracy: The extracted polylines should closely follow the true centerlines or boundaries of roads, with minimal geometric distortion or displacement.

Topological Correctness: The network must preserve connectivity, intersections, and the correct branching structure, ensuring that roads are not fragmented or erroneously merged.

Suitability for Downstream Applications: The resulting polylines should be compatible with routing engines, map generation workflows, and GIS systems, supporting tasks such as shortest-path computation, map updating, and attribute assignment.

Robustness: The pipeline must handle real-world challenges, including noise, occlusions (e.g., from trees or vehicles), and fragmented or missing segments.

Scalability and Automation: The process should be efficient and scalable to large datasets, minimizing manual intervention.

These criteria are reflected in both pixel-level metrics (e.g., IoU, F1-score) and topology-aware, graph-level metrics (e.g., Average Path Length Similarity, TOPO F1).

Preprocessing Binary and Multi-Class Masks

Binarization and Normalization

Instance segmentation models output probability maps or soft masks, which must be thresholded to obtain binary masks. The choice of threshold can significantly affect the quality of the extracted network; adaptive thresholding or Otsu's method may be used for images with varying illumination. For multi-class masks (e.g., distinguishing roads from other features), each class is typically binarized separately.

Normalization of input images and masks is also crucial, especially when integrating data from multiple sources or resolutions. Standardizing pixel values and spatial resolution ensures consistency across the pipeline.

Morphological Cleaning

Raw masks often contain noise, small artifacts, and holes. Morphological operations—such as opening (erosion followed by dilation), closing (dilation followed by erosion), and area filtering—are applied to remove small spurious regions and fill gaps. The choice of structuring element (e.g., disk, square) and its size should be tuned to the expected road width and image resolution.

For example, in the Massachusetts Roads dataset, morphological opening and closing are used to clean up the mask before skeletonization, ensuring that only contiguous road regions are retained.

Data Augmentation and Occlusion Simulation

To improve robustness to occlusions and noise, data augmentation techniques such as random cropping, flipping, rotation, and synthetic occlusion (e.g., masking out patches) are employed during training. Class-specific noise injection, where noise is added preferentially to the largest or most frequent classes, can further enhance model generalization to real-world conditions with occlusions and variable lighting.

Skeletonization Fundamentals and Classical Algorithms

Skeletonization is the process of reducing a binary mask to its topological skeleton—a one-pixel-wide representation that preserves the connectivity and branching structure of the original shape. This step is critical for extracting road centerlines from thick masks.

Iterative Thinning Algorithms

The most widely used skeletonization methods are iterative thinning algorithms, which iteratively remove boundary pixels while preserving connectivity:

Zhang-Suen Thinning: A parallel thinning algorithm that removes pixels in two sub-iterations, ensuring that the skeleton remains connected and one-pixel wide.

Guo-Hall Thinning: Similar to Zhang-Suen but with improved endpoint detection and parallelization.

Lee's Method: Designed for 3D images but applicable to 2D; uses octree data structures for efficient thinning.

These algorithms are implemented in libraries such as scikit-image (skeletonize), OpenCV, and DIPlib.

Medial Axis Transform

The medial axis transform computes the set of points equidistant from the boundaries of the shape, providing both the skeleton and a distance map that encodes local width. While the medial axis can produce more accurate centerlines, it is sensitive to noise and may generate spurious branches.

Maximal Disk and Straight Skeleton Methods

Maximal Disk (MD): Identifies centers of maximal inscribed circles within the shape, forming the skeleton. Efficient and robust for polygonal shapes, but may require additional processing to handle ligatures and curved segments.

Straight Skeleton: Constructs skeletons by simulating the inward movement of polygon edges, producing straight-line branches. Useful for vector data and applications such as park road network generation.

Comparison of Skeletonization Algorithms

Algorithm

Pros

Cons

Zhang-Suen

Preserves connectivity, efficient

May not always yield unit-pixel width

Guo-Hall

Good endpoint detection, parallelizable

Can produce noisy branches

Medial Axis

Encodes width, accurate centerline

Sensitive to noise, spurious branches

Maximal Disk

Efficient, robust for polygons

May lose accuracy in curved/ligature regions

Straight Skeleton

Good for vector data, interpretable

Limited to polygons, less flexible

Skeletonization is often followed by pruning to remove short spurs and artifacts caused by boundary irregularities.

Thinning, Pruning, and Skeleton Refinement Techniques

Thinning

Thinning reduces the width of lines in the mask to a single pixel, preserving the overall structure. It is typically performed using the aforementioned iterative algorithms. The number of iterations and the choice of structuring elements can be tuned to balance between completeness and over-thinning.

Pruning

Pruning removes short, spurious branches (spurs) from the skeleton, which often arise from noise or minor boundary irregularities. This is achieved by:

Identifying endpoints and branch points in the skeleton.

Removing branches shorter than a specified length threshold.

Optionally, applying conditional dilation to restore legitimate endpoints after pruning.

Pruning is essential for ensuring that the skeleton represents only meaningful road segments and intersections.

Skeleton Refinement

Refinement may include:

Smoothing: Applying mean or Gaussian filtering to the skeleton coordinates to reduce jaggedness.

Conditional Dilation: Restoring endpoints that may have been inadvertently removed during pruning.

Topology Correction: Ensuring that the skeleton remains connected and that intersections are preserved.

Distance Transform and Centerline Extraction Methods

The distance transform computes, for each pixel in the mask, the distance to the nearest boundary. This information is used for:

Centerline Extraction: Identifying the ridge (local maxima) of the distance map, which corresponds to the centerline of the road.

Width Estimation: The value of the distance transform at the centerline provides an estimate of the local road width.

Centerline extraction via distance transform is robust to variations in road width and can be combined with skeletonization for improved accuracy.

Learned Skeletonization and End-to-End Vectorization Networks

Recent advances in deep learning have led to the development of end-to-end networks that directly predict skeletons or vectorized representations from images:

RoadVecNet: Interlinked U-Net networks for simultaneous road segmentation and vectorization, using a Sobel operator for smooth centerline extraction.

DeepCenterline: Multi-task FCN that predicts both a centerline distance map and endpoint confidence map, followed by minimal path extraction for robust skeletonization.

SAM-Road and RNGDet++: Adaptations of the Segment Anything Model (SAM) and transformer-based architectures for direct graph extraction, predicting vertices and edges without extensive post-processing.

Learned Skeletonization (U-Net, RCNN-UNet): Networks trained to output one-pixel-wide skeletons, often with specialized loss functions to enforce connectivity and thinness.

These approaches often incorporate topology-aware losses, multi-scale feature extraction, and connectivity supervision to improve the quality of the extracted network.

Raster-to-Vector Conversion Algorithms and Libraries

Once a clean skeleton is obtained, the next step is to convert it into vector polylines:

Contour Tracing: Extracts the boundaries or centerlines as sequences of connected pixels, which are then converted to polylines.

Graph Construction: Builds a graph where nodes correspond to junctions or endpoints, and edges correspond to road segments.

Polyline Simplification: Reduces the number of points in the polyline while preserving shape, using algorithms such as Ramer–Douglas–Peucker (RDP) or Visvalingam–Whyatt.

Popular libraries and tools include:

scikit-image: skeletonize, medial_axis, and thin for skeletonization; find_contours for contour extraction.

OpenCV: Morphological operations, contour detection, and vectorization.

osmnx: For converting raster/skeleton to graph structures and integrating with OSM data.

GRASS GIS: r.thin and r.to.vect for thinning and vectorization.

ArcScan (ArcGIS): Automated vectorization of raster maps, with parameters for gap closure and hole size.

Google Earth Engine: reduceToVectors for raster-to-vector conversion.

Polyline Simplification, Smoothing, and Geometric Fitting

Polyline simplification is essential for reducing the complexity of the extracted network and ensuring geometric fidelity:

Ramer–Douglas–Peucker (RDP): Recursively removes points that are within a specified distance (epsilon) from the simplified line, preserving key shape features.

Visvalingam–Whyatt: Removes points with the smallest effective area, prioritizing the preservation of significant bends and features.

Least Squares Fitting: Fits lines or curves to the polyline segments for further smoothing.

Spline Fitting: Fits smooth curves (e.g., B-splines) to the polyline for applications requiring high geometric accuracy.

The choice of simplification algorithm and parameters should balance between reducing noise and preserving critical geometric and topological features, especially at intersections and sharp bends.

Graph Construction from Polylines and Masks

After vectorization, the polylines are organized into a graph structure:

Nodes: Represent intersections, endpoints, or junctions. Detected by identifying points with degree ≥ 3 (branch points) or degree 1 (endpoints) in the skeleton.

Edges: Correspond to road segments between nodes. Each edge is a polyline with associated attributes (e.g., length, width, class).

Attributes: Additional information such as road width (from distance transform), class (from mask), or speed limits (from external data) can be assigned to nodes and edges.

Graph construction is facilitated by libraries such as NetworkX, osmnx, and GIS software. The resulting graph can be exported in standard formats (e.g., GeoJSON, Shapefile) for further analysis or integration.

Junction/Intersection Detection and Node Snapping Strategies

Accurate detection of intersections and proper snapping of nodes are critical for topological correctness:

Intersection Detection: Identifies points where multiple skeleton branches meet. Kernel Density Estimation (KDE) on connecting points or analysis of node degree can be used.

Node Snapping: Ensures that nodes at intersections are precisely aligned, especially when merging fragmented segments or integrating with external data (e.g., OSM). Snapping can be based on spatial proximity (within a tolerance) or attribute similarity.

Cluster Processing: In GIS systems like ArcGIS, cluster tolerance is used to merge vertices within a specified distance, ensuring geometric integration and avoiding duplicate nodes.

Proper handling of intersections is essential for routing applications and for maintaining the integrity of the road network.

Topology Validation, Planar Graph Constraints, and Correction

Topology validation ensures that the extracted network is free of errors such as disconnected segments, overlaps, or invalid intersections:

Planar Graph Constraints: The network should be planar (no crossing edges except at intersections) and connected. Planar augmentation and constrained triangulation can be used to validate and correct planar partitions.

Topology Rules: GIS systems enforce rules such as "must not overlap," "must connect at endpoints," and "must not have gaps".

Error Correction: Detected errors (e.g., gaps, overlaps) can be corrected by snapping, merging, or splitting features as needed.

Automated topology validation is supported in tools like ArcGIS, QGIS, and custom scripts using NetworkX or other graph libraries.

Handling Fragmented Segments: Gap Detection and Linking

Fragmented segments arise from occlusions, noise, or segmentation errors. Gap detection and linking strategies include:

Connected Component Analysis: Identifies disconnected regions in the mask or skeleton. Small isolated components can be removed as noise, while larger fragments may be candidates for linking.

Geometric Linking: Computes the shortest distance between endpoints of fragmented segments. If the distance is below a threshold, a connecting edge is added.

Graph Augmentation: Adds edges to ensure biconnectivity or to restore missing connections, as in planar augmentation methods.

Morphological Bridging: Dilation followed by thinning can bridge small gaps before skeletonization.

Careful parameter tuning is required to avoid introducing false connections.

Handling Occlusions and Missing Data

Occlusions from trees, vehicles, or shadows are common in real-world imagery. Strategies for handling occlusions include:

Data Augmentation: Training with synthetic occlusions to improve model robustness.

Super-Resolution: Enhancing image resolution to recover fine details obscured by occlusions.

Inpainting: Filling missing regions using context from surrounding pixels or multi-view imagery.

Multi-Modal Fusion: Combining data from multiple sensors (e.g., optical, SAR) or time points to recover occluded segments.

Post-Processing: Reconstructing broken road lines based on geometric proximity of connected domains or convex hulls.

Noise Robustness: Denoising, Morphological Filtering, CRF/Graph Cuts

Noise in segmentation masks can lead to spurious branches, false positives, or fragmented roads. Robustness is improved by:

Morphological Filtering: Opening, closing, and area filtering to remove small artifacts.

Conditional Random Fields (CRF): Post-processing with CRFs to enforce spatial consistency and smooth boundaries.

Graph Cuts: Energy-based segmentation to refine boundaries and suppress noise.

Edge-Preserving Filtering: Filters such as bilateral or Gabor filters to retain road edges while reducing noise.

Multi-Scale and Multi-Resolution Processing Strategies

Road networks exhibit features at multiple scales, from highways to narrow alleys. Multi-scale processing includes:

Pyramid Pooling: Aggregates features at different spatial scales to capture both global context and local details.

Atrous/Dilated Convolutions: Expands the receptive field without increasing parameters, enabling detection of long-range dependencies.

Multi-Branch Networks: Parallel convolutional branches with different kernel sizes to capture features at various scales.

Multi-scale strategies improve the extraction of both major and minor roads, as well as the handling of complex intersections.

Topology-Aware Training Losses and Connectivity Supervision

Standard pixel-wise losses (e.g., cross-entropy, Dice) do not enforce topological correctness. Topology-aware losses include:

Connectivity Loss: Penalizes disconnected predictions, encouraging the network to produce continuous road segments.

Topology-Aware Loss (TC-Loss): Aligns the persistence fields and critical points between prediction and ground truth, directly regulating topological invariants.

Adversarial Losses: Discriminators trained to distinguish between real and predicted road networks, enforcing structural realism.

Multi-Task Losses: Jointly optimize for segmentation, centerline extraction, and orientation prediction.

Connectivity supervision is often implemented by providing explicit connectivity labels or by augmenting the loss function with topological constraints.

Post-Processing for Routing Suitability and Map Generation

The final polylines must be suitable for routing and map generation:

Graph Cleaning: Removes duplicate edges, merges colinear segments, and ensures that all nodes are properly connected.

Attribute Assignment: Assigns attributes such as road class, width, and speed limits, either from the mask or by integrating external data.

Export to GIS Formats: Outputs the network in standard formats (e.g., Shapefile, GeoJSON, OSM XML) for integration with GIS and routing engines.

Validation: Ensures that the network supports routing queries, with all intersections and endpoints correctly represented.

Attribute Extraction from Masks

Beyond geometry and topology, additional attributes can be extracted:

Width: Estimated from the distance transform at the centerline or from mask statistics.

Class: Derived from multi-class masks or by integrating with external data sources.

Speed Limits: Inferred from road class or by matching with OSM or other databases.

Attribute extraction enhances the utility of the network for navigation, simulation, and analysis.

Evaluation Metrics and Topology-Aware Benchmarks

Evaluation of mask-to-polyline conversion requires both pixel-level and graph-level metrics:

Pixel-Level Metrics: IoU, F1-score, precision, recall, accuracy.

Graph-Level Metrics:

Average Path Length Similarity (APLS): Compares shortest path lengths between node pairs in the predicted and ground truth graphs, emphasizing connectivity and routing suitability.

TOPO F1: Measures edge-level precision and recall, accounting for spatial tolerance in edge matching.

Relaxed Precision/Recall: Allows for small spatial deviations in matching.

Hausdorff and Fréchet Distance: Quantifies geometric similarity between polylines.

APLS and TOPO metrics are particularly important for applications where routing and network connectivity are critical.

Datasets and Benchmarks for Training and Evaluation

Several public datasets support training and evaluation of road extraction and vectorization pipelines:

SpaceNet Roads: High-resolution satellite imagery with road centerline annotations, used in the SpaceNet challenge.

DeepGlobe Road Extraction: 6226 satellite image tiles with binary road masks, covering diverse geographic regions.

Massachusetts Roads: 1171 aerial images with binary masks, widely used for benchmarking segmentation and vectorization methods.

Ottawa Road Imagery: Google Earth images with high-resolution road annotations.

Cityscapes: Urban street scenes with multi-class segmentation, useful for evaluating robustness to occlusions and complex backgrounds.

These datasets provide ground truth for both pixel-level and graph-level evaluation.

Open-Source Tools, Libraries, and GIS Software

A range of open-source tools support mask-to-polyline conversion:

scikit-image: Skeletonization, medial axis, contour extraction, and morphological operations.

OpenCV: Morphological filtering, contour detection, and vectorization.

osmnx: Graph construction, integration with OSM, and routing analysis.

GRASS GIS: Thinning and vectorization modules.

ArcScan (ArcGIS): Automated raster-to-vector conversion with customizable parameters.

GAMA Platform: Network cleaning and optimization, including gap closure and intersection splitting.

Sat2Graph: Evaluation toolkit for APLS and TOPO metrics.

OSRM, GraphHopper: Routing engines that require topologically correct vector networks.

These tools can be integrated into custom pipelines for scalable, automated processing.

Specialized Research Systems and Recent Papers

Recent research has advanced the state of the art in mask-to-polyline conversion:

SAM-Road: Adapts the Segment Anything Model for direct graph extraction, achieving high accuracy and speed.

RNGDet++: Transformer-based network with instance segmentation and multi-scale feature enhancement for road network graph detection.

RoadVecNet: Interlinked U-Net networks for joint segmentation and vectorization.

DeepCenterline: Multi-task FCN for robust centerline extraction with minimal path post-processing.

MRENet, RISENet: Multi-task and attention-based networks for simultaneous road surface and centerline extraction, with strong performance on challenging datasets.

These systems often release code and pretrained models, facilitating reproducibility and benchmarking.

Integration with OpenStreetMap and Map Updating Workflows

Integration with OSM requires:

Georeferencing: Ensuring that extracted polylines are accurately aligned with geographic coordinates.

Attribute Matching: Assigning OSM-compatible tags (e.g., highway type, surface) to extracted features.

Conflict Resolution: Merging new data with existing OSM features, handling overlaps and discrepancies.

Map Matching: Aligning GPS trajectories or extracted polylines with the OSM network for validation or updating.

Automated tools and scripts can facilitate the updating of OSM with newly extracted road networks.

Routing Engines and Downstream Validation

Routing engines such as OSRM and GraphHopper require:

Edge-Expanded Graphs: Conversion of polylines into edge-based graphs with turn restrictions and weights.

Attribute Assignment: Assigning speed limits, access restrictions, and other routing-relevant attributes.

Validation: Ensuring that the network supports routing queries, with all intersections and endpoints correctly represented.

Downstream validation includes running routing queries and comparing results with ground truth or existing maps.

Production Pipelines, Performance, and Scalability Considerations

Scalable production pipelines must address:

Batch Processing: Efficient handling of large datasets, possibly in parallel or distributed environments.

Profiling and Optimization: Identifying bottlenecks and optimizing key stages (e.g., skeletonization, vectorization).

Automation: Minimizing manual intervention through robust parameter selection and error handling.

Monitoring and Logging: Tracking performance, errors, and data quality across the pipeline.

Best practices include modular design, use of open-source libraries, and integration with cloud or HPC resources for large-scale processing.

Comparison Table: Vectorization and Refinement Techniques

Technique

Description

Tools/Libraries

Strengths

Limitations

Skeletonization (Thinning)

Reduces binary mask to 1-pixel wide centerlines

scikit-image, OpenCV

Preserves topology, simplifies structure

Sensitive to noise, may produce spurious branches

Medial Axis Transform

Computes centerline and local width

scikit-image

Encodes width, accurate centerline

Sensitive to noise, spurious branches

Maximal Disk Skeletonization

Centers of maximal inscribed circles

Custom, scikit-geometry

Efficient, robust for polygons

May lose accuracy in curved/ligature regions

Straight Skeleton

Inward movement of polygon edges

scikit-geometry

Good for vector data, interpretable

Limited to polygons, less flexible

Morphological Filtering

Removes noise and fills gaps using morphological operations

OpenCV, scikit-image

Simple, fast, effective for small artifacts

Limited in handling complex noise or occlusions

Ramer–Douglas–Peucker (RDP)

Simplifies polylines by reducing number of points

rdp (Python), OpenCV

Reduces complexity, preserves shape

May oversimplify fine details

Visvalingam–Whyatt

Removes points with smallest effective area

Custom, GIS software

Preserves significant bends

May remove important features if not tuned

GRASS GIS r.thin + r.to.vect

Thinning and vectorization of raster lines

GRASS GIS

GIS-integrated, robust for large-scale data

Requires GIS setup, less flexible

osmnx

Converts raster/skeleton to graph structures

osmnx

Graph-based, integrates with OSM

May require tuning for sparse/fragmented data

ArcScan (ArcGIS)

Automated raster-to-vector conversion

ArcGIS

Efficient, customizable parameters

Proprietary, requires license

Sat2Graph

Evaluation toolkit for APLS and TOPO metrics

Sat2Graph

Standardized evaluation

Requires format conversion

Each technique should be selected and tuned based on the specific requirements of the application, data characteristics, and available computational resources.

Conclusion

Restoring clean, accurate polylines from instance segmentation masks is a multi-stage process that integrates classical image processing, advanced deep learning, and graph theory. The pipeline—from preprocessing and skeletonization to vectorization, graph construction, and post-processing—must be carefully designed to ensure geometric fidelity, topological correctness, and suitability for downstream applications such as map generation and routing.

Best practices include robust morphological cleaning, the use of topology-preserving skeletonization algorithms, careful pruning and simplification, and rigorous topology validation. Recent advances in deep learning offer promising end-to-end solutions, but classical methods remain essential for post-processing and refinement.

Evaluation should combine pixel-level and graph-level metrics, with a focus on connectivity and routing suitability. Open-source tools and public datasets facilitate reproducibility and benchmarking, while integration with GIS and routing engines enables practical deployment.

As the field advances, continued research into topology-aware learning, multi-modal data fusion, and scalable automation will further enhance the accuracy and utility of extracted road networks, supporting a wide range of geospatial and navigation applications.

References (40)

A dynamic attention mechanism for road extraction from high ... - Nature. https://www.nature.com/articles/s41598-025-02267-6

MRENet: Simultaneous Extraction of Road Surface and Road Centerline in .... https://www.mdpi.com/2072-4292/13/2/239

Data Preprocessing Pipeline | Project-OSRM/osrm-backend | DeepWiki. https://deepwiki.com/Project-OSRM/osrm-backend/2-data-preprocessing-pipeline

Understanding OSRM Graph Representation - GitHub. https://github.com/Telenav/open-source-spec/blob/master/osrm/doc/understanding_osrm_graph_representation.md

SpaceNet 3: Road Network Detection. https://spacenet.ai/spacenet-roads-dataset/

APLS: Average Path Length Similarity | adinh26101. https://adinh26101.github.io/en/posts/apls/

Evaluation Metrics | earth-insights/samroadplus | DeepWiki. https://deepwiki.com/earth-insights/samroadplus/7.2-evaluation-metrics

TopoRF-Net: Topology-Aware Road Segmentation in Multi-Resolution Remote .... https://www.mdpi.com/1424-8220/25/24/7428

Extraction and Calculation of Roadway Area from Satellite Images Using .... https://www.mdpi.com/2313-433X/8/5/124

Lecture11 - University of Technology, Iraq. https://uotechnology.edu.iq/ce/lecture 2013n/4th Image Processing _Lectures/DIP_Lecture11.pdf

Segmentation Cleanup - Muddling through Medical Imaging. https://salcedoe.github.io/MtMdocs/imageProcessing/ImageSegmentationCleanup/

Refine Segmentation Using Morphology in Image Segmenter. https://www.mathworks.com/help/images/use-morphological-techniques-to-refine-a-segmentation.html

Class-Specific Noise Injection for Improved Road Segmentation. https://link.springer.com/chapter/10.1007/978-3-031-71716-1_8

Morphology - Thinning - University of Edinburgh. https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

Skeleton Generation for Digital Images Based on Performance Evaluation .... http://article.nadiapub.com/IJSIP/vol9_no2/5.pdf

Skeletonize — skimage 0.26.0 documentation - scikit-image. https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html

SKELETON-BASED AUTOMATIC ROAD NETWORK EXTRACTION FROM AN ORTHOPHOTO .... https://www.researchgate.net/profile/Elyta-Widyaningrum/publication/337398825_SKELETON-BASED_AUTOMATIC_ROAD_NETWORK_EXTRACTION_FROM_AN_ORTHOPHOTO_COLORED_POINT_CLOUD/links/5dd549ad458515cd48afb9b4/SKELETON-BASED-AUTOMATIC-ROAD-NETWORK-EXTRACTION-FROM-AN-ORTHOPHOTO-COLORED-POINT-CLOUD.pdf

Performance Comparison of ZS and GH Skeletonization Algorithms. https://research.ijcaonline.org/volume121/number24/pxc3905138.pdf

Road Intersection Detection through Finding Common Sub-Tracks ... - MDPI. https://www.mdpi.com/2220-9964/6/10/311

Research on Automatic Generation of Park Road Network Based on Skeleton .... https://www.mdpi.com/2076-3417/14/18/8475

Lane Centerline Extraction Based on Surveyed Boundaries: An ... - MDPI. https://www.mdpi.com/1424-8220/25/8/2571

arXiv:1903.10481v1 [cs.CV] 25 Mar 2019. https://arxiv.org/pdf/1903.10481

Road Width Estimator—An Automatic Tool for Calculating Road Width .... https://link.springer.com/article/10.1007/s41651-024-00205-0

RNGDet++: Road Network Graph Detection by Transformer with Instance .... https://arxiv.org/abs/2209.10150

GitHub - TonyXuQAQ/RNGDetPlusPlus: [RAL 2023] RNGDet++: Road Network .... https://github.com/TonyXuQAQ/RNGDetPlusPlus

Ramer–Douglas–Peucker algorithm - Wikipedia. https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm

Polyline Simplification - matthewdeutsch.com. https://www.matthewdeutsch.com/projects/polyline-simplification/

VE-GCN: A Geography-Aware Approach for Polyline Simplification in .... https://www.mdpi.com/2220-9964/14/2/64

Building a Road Extraction Model with Convolutional Neural Networks and .... https://mapsandlocations.com/building-a-road-extraction-model-with-convolutional-neural-networks-and-satellite-data/

GitHub - cyang-kth/osm_mapmatching: A tutorial on map matching using .... https://github.com/cyang-kth/osm_mapmatching

Raster to Vector Conversion - Google Developers. https://developers.google.com/earth-engine/guides/reducers_reduce_to_vectors

Topology in ArcGIS—ArcGIS Pro | Documentation - Esri. https://pro.arcgis.com/en/pro-app/latest/help/data/topologies/topology-in-arcgis.htm

Polyline Drawings with Topological Constraints. https://arxiv.org/pdf/1809.08111.pdf

VALIDATION OF PLANAR PARTITIONS USING CONSTRAINED TRIANGULATIONS - gdmc.nl. https://gdmc.nl/publications/2010/Validation_planar_partitions.pdf

Efficient Occluded Road Extraction from High-Resolution Remote ... - MDPI. https://www.mdpi.com/2072-4292/13/24/4974

Road segmentation made simple: a practical comparison of segmentation .... https://www.nrso.ntua.gr/geyannis/wp-content/uploads/geyannis-pc603.pdf

TopoAL: An Adversarial Learning Approach for Topology-Aware Road .... https://arxiv.org/abs/2007.09084

Example usage of clean_network in GAMA Platform. https://gama-platform.org/wiki/CleaningGISData

SpaceNet - Registry of Open Data on AWS. https://registry.opendata.aws/spacenet/

Mastering Pipeline Performance Optimization - Data Engineering. https://dataengineering.blog/mastering-pipeline-performance-optimization/