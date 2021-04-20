# >>> Must-read papers on graph neural networks (GNN)
# (C) https://www.facebook.com/.../permalink/492800285460365/
# Trong năm 2020, cùng với Transformer, GNN hay Graph Neural Network nhận được nhiều sự chú ý và quan tâm hơn từ cộng đồng. 1 số lớp bài toán điển hình của GNN bao gồm: Node Classification, Graph Classification, Link Prediction, Graph Clustering,...
# Trong thực tế, GNN có thể được áp dụng vào nhiều kiểu dữ liệu và bài toán khác nhau. 1 số ứng dụng như:
# - https://arxiv.org/abs/2102.07753 dùng CNN với GNN phía sau cho metric learning để học embedding, có so sánh với rất nhiều các pp khác từ triplet loss, angular loss,.. và là phương pháp SOTA hiện tại
# - https://arxiv.org/abs/2003.07493 : paper này thì dùng graph với bài toán scene text detection
# - https://arxiv.org/abs/2002.01276 : paper này thì dùng GCN với CTC loss cho bài toán scene text recognition
# - hay ví dụ 1 bài toán đang khá nổi bật trong thời gian gần đây như: Scene Graph Generation. 1 số bài toán khác như keypoint matching (SuperGlue) https://arxiv.org/abs/1911.11763 , hand-pose estimation (HOPE): https://arxiv.org/abs/2004.00060 , information extraction từ ảnh (PICK): https://arxiv.org/abs/2004.07464 .
# - task về Image-Text Matching, phục vụ cho bài toán như Visual Semantic Search Engine: https://arxiv.org/abs/1909.02701
# - task về session-based recommend system: https://arxiv.org/abs/1811.00855
# - task về text classification (TextGCN): https://arxiv.org/abs/1809.05679 , 1 paper khác áp dụng GCN cho bài toán phân loại văn bản tiếng Việt: https://www.researchgate.net/.../344433020_Vietnamese...
# - task về table structure recognition, lưu giữ định dạng bảng biểu: (Rethinking TSN) https://arxiv.org/abs/1905.13391 , (Complicated TSN) https://arxiv.org/abs/1908.04729
# - GCN áp dụng trong bài toán Multi-Object Tracking: https://arxiv.org/abs/2006.13164
# - GCN còn được sử dụng trong y sinh, biểu diễn thuốc, DNA dưới dạng graph, 1 paper kèm theo của tác giả Tuan Nguyen (tác giả ebook Deep Learning cơ bản) https://ieeexplore.ieee.org/document/9359501
# - và còn rất nhiều các ứng dụng khác nữa...
# Tham khảo tại: https://github.com/thunlp/GNNPapers
# (C) https://mobile.twitter.com/oma.../status/1368167852763717641
# 1 số bài chia sẻ khác trên Viblo về GNN
# - Tổng quan về GNN: https://viblo.asia/p/6J3ZgP0qlmB
# - GNN cho bài toán trích rút thông tin từ ảnh hóa đơn: https://viblo.asia/p/djeZ1yPGZWz
# #Q12021 #viblo #machine_learning #deep_learning #gnn #gcn #graph_neural_network #graph_convolutional_network