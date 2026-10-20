UTS Graf Sains Data: Neo4j, Graph Data Science, dan Rekomendasi KNN
Nama: Bima Sena
NIM: 24031554214
Program Studi: S1 Sains Data, FMIPA, Universitas Negeri Surabaya
Mata Kuliah: Graf Sains Data (UTS Gasal 2026/2027)

Repositori ini berisi pengerjaan UTS Graf Sains Data. Dataset yang digunakan adalah RetailRocket (Roman Zykov, Kaggle), yaitu catatan interaksi pengunjung toko online berupa view, addtocart, dan transaction. Data dimasukkan ke Neo4j sebagai graf Visitor dan Item, kemudian dianalisis dengan vector space modelling (VSM), proyeksi graf, Node Similarity pada Graph Data Science (GDS), dan rekomendasi KNN yang dievaluasi dengan Hit@5 dan NDCG@5.

Dokumentasi lengkap beserta kode dan output ada pada notebook UTS_GSD.ipynb. README ini merangkum arsitektur, cara menjalankan, dan jawaban setiap soal.

Periode Data
Periode	Rentang	Fungsi	Jumlah baris
P1	9 Mei 2015 sampai 26 Juni 2015 (49 hari)	Training, dimasukkan ke Neo4j	1.001.321
P2	27 Juni 2015 sampai 5 Agustus 2015 (40 hari)	Testing, ground truth evaluasi	868.676
Kedua periode tidak tumpang tindih. Filter diterapkan pada kolom timestamp (unix epoch, satuan milidetik).

Struktur Folder
UTS_graph/
├── UTS_GSD.ipynb          notebook utama (kode, output, dan penjelasan)
├── import_data.py         skrip impor data ke Neo4j (dari soal bagian 1.3)
├── tutorial_neo4j.py      latihan tutorial driver Neo4j (bagian 1.2)
├── requirements.txt       dependensi Python
├── sebelum.png            graf sebelum perhitungan similarity
├── sesudah.png            graf sesudah perhitungan similarity
├── data/                  dataset asli RetailRocket (events.csv)
├── data_p1/               events.csv hasil filter P1 (training)
├── data_p2/               events.csv hasil filter P2 (testing)
└── myvenv/                virtual environment (tidak ikut diunggah)
Berkas ecommerce-dataset.zip, folder myvenv, dan .ipynb_checkpoints tidak perlu diunggah ke GitHub (lihat .gitignore). Jika data/events.csv melebihi batas 100 MB per berkas GitHub, cukup unggah data_p1 dan data_p2.

Arsitektur Pengerjaan

Model graf di Neo4j:

(:Visitor {visitorId}) -[:VIEWED {timestamp}]-------> (:Item {itemId})
(:Visitor {visitorId}) -[:ADDED_TO_CART {timestamp}]-> (:Item {itemId})
(:Visitor {visitorId}) -[:BOUGHT {timestamp, transactionId}]-> (:Item {itemId})
(:Visitor) -[:SIMILAR_TO {similarity}]-> (:Visitor)      dibuat pada nomor 7b
Cara Menjalankan
Prasyarat: Python 3.10, Neo4j Desktop (database aktif di neo4j://127.0.0.1:7687) dengan plugin Graph Data Science terpasang, dan dataset RetailRocket.

Buat dan aktifkan virtual environment.

python -m venv myvenv
# Windows
myvenv\Scripts\activate
# Linux / macOS
source myvenv/bin/activate
Pasang dependensi.

pip install -r requirements.txt
pip install pandas numpy scipy networkx matplotlib tqdm jupyter
Unduh dataset dari Kaggle lalu letakkan events.csv di folder data/.

curl -L -o ecommerce-dataset.zip https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset
Pastikan database Neo4j kosong. Jalankan di Neo4j Browser bila perlu:

MATCH (n) DETACH DELETE n;
Buka UTS_GSD.ipynb lalu jalankan seluruh cell dari atas ke bawah. Password Neo4j diminta lewat getpass, jadi tidak tersimpan di notebook. Pada tahap impor, notebook menjalankan:

python import_data.py -u neo4j -p <password> -s data_p1
Untuk gambar nomor 8, jalankan dua query pada bagian jawaban nomor 8 di Neo4j Browser, lalu simpan hasilnya sebagai sebelum.png dan sesudah.png.

Catatan: cell penulisan SIMILAR_TO (7b) hanya boleh dijalankan sekali, karena menjalankannya ulang menambah relasi ganda.

Jawaban Soal
2.1 Wrangling dan EDA
Nomor 1 dan 2: periode dan filter. P1 (9 Mei sampai 26 Juni 2015) dan P2 (27 Juni sampai 5 Agustus 2015) difilter pada timestamp. Batas akhir ditambah satu hari dikurangi satu milidetik agar hari terakhir ikut terhitung. Hasilnya disimpan ke data_p1/events.csv (view 968.775, addtocart 24.500, transaction 8.046) dan data_p2/events.csv (view 839.843, addtocart 21.658, transaction 7.175).

Nomor 3: komposisi subgraph. Subgraph P1 terdiri atas 509.883 simpul Visitor dan 149.658 simpul Item (total 659.541), dengan 1.001.321 relasi: VIEWED 968.775, ADDED_TO_CART 24.500, dan BOUGHT 8.046. Query yang digunakan:

MATCH (n)
UNWIND labels(n) AS jenis
RETURN jenis, count(*) AS jumlah
ORDER BY jumlah DESC
MATCH (n) mengambil semua simpul, labels(n) mengembalikan daftar label, UNWIND memecahnya menjadi satu baris per label, dan count(*) menghitung tiap label (pengelompokan otomatis berdasarkan kolom non-agregat jenis).

MATCH (a)-[r]->(b)
RETURN labels(a)[0] AS dari, type(r) AS relasi, labels(b)[0] AS ke, count(*) AS jumlah
ORDER BY jumlah DESC
Pola (a)-[r]->(b) mencocokkan setiap relasi berarah, type(r) mengambil jenis relasi, dan hasilnya dihitung per kombinasi asal, relasi, dan tujuan.

Nomor 4: VSM user. Matriks user x item berukuran 509.883 x 149.658 dengan 773.006 elemen tidak nol, density sekitar 1,01 x 10^-5 (sparsity 99,999 persen), sehingga disimpan sebagai matriks sparse (CSR). Rata-rata satu user berinteraksi dengan 1,52 item, median 1, dan 407.118 user (79,85 persen) hanya berinteraksi dengan satu item. Plot networkx 10 user contoh membentuk 10 kelompok yang tidak saling terhubung.

Nomor 5: VSM item. Matriks item x user (transpose) berukuran 149.658 x 509.883 dengan sparsity yang sama. Rata-rata satu item disentuh 5,17 user, median 2, dan 65.900 item (44,03 persen) hanya disentuh satu user. Sebaran tetap miring ke kanan, tetapi tidak sepincang sisi user.

Nomor 6: proyeksi yang lebih baik.

user-user	item-item
Jumlah simpul	509.883	149.658
Simpul yang punya tetangga	466.825 (91,56%)	87.944 (58,76%)
Pasangan relasi (batas atas)	14.543.250	7.575.346
Proyeksi item-item lebih efisien (lebih kecil), tetapi proyeksi user-user lebih efektif karena 91,56 persen simpulnya punya tetangga, sedangkan pada item-item 41,24 persen item terisolasi. Proyeksi yang dipilih adalah user-user, karena tujuan 2.2 adalah merekomendasikan item kepada user melalui user-user yang mirip.

Nomor 7: similarity. Similarity dihitung dengan Node Similarity (Jaccard) pada GDS versi 2026.09.0 dari graf graf_user_item (659.541 simpul, 1.001.321 relasi). User acuan dipilih dengan random.seed(2024) dari user dengan 5 sampai 30 item: 1084423 (13 item).

7a. Lima user paling mirip:

MATCH (acuan:Visitor {visitorId: $id})
CALL gds.nodeSimilarity.filtered.stream('graf_user_item', {
    sourceNodeFilter: acuan,
    topK: 5
})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).visitorId AS user_acuan,
       gds.util.asNode(node2).visitorId AS user_mirip,
       similarity
ORDER BY similarity DESC
user_mirip	similarity
1292597	0,153846
1098746	0,153846
417861	0,142857
796888	0,125
30358	0,125
7b. Nilai similarity ditulis ke graf sebagai relasi SIMILAR_TO (properti similarity) dengan gds.nodeSimilarity.filtered.write, lalu dibaca kembali dengan query biasa. Hasilnya 1 simpul dibandingkan dan 5 relasi ditulis, sama dengan hasil 7a.

Nomor 8: sebelum dan sesudah similarity. Jumlah relasi sebelum: ADDED_TO_CART 24.500, BOUGHT 8.046, VIEWED 968.775. Sesudah ditambah SIMILAR_TO sebanyak 5, sehingga total 1.001.326 relasi.

Sebelum (hanya relasi VIEWED dari user ke item bersama, tanpa garis antar user):

MATCH (a:Visitor {visitorId: 1084423})-[r1]->(i:Item)<-[r2]-(b:Visitor)
WHERE b.visitorId IN [1292597, 1098746, 417861, 796888, 30358]
RETURN a, r1, i, r2, b
Graf sebelum perhitungan similarity

Sesudah (muncul garis SIMILAR_TO dari user acuan 1084423 ke lima user yang mirip):

MATCH (a:Visitor {visitorId: 1084423})-[s:SIMILAR_TO]->(b:Visitor)
OPTIONAL MATCH (a)-[r1]->(i:Item)<-[r2]-(b)
RETURN a, s, b, r1, i, r2
Graf sesudah perhitungan similarity

2.2 Analytics
Definisi yang dipakai: KNN berbasis user dengan kemiripan Jaccard dari item yang disentuh pada P1 dan k = 5. Setiap item milik tetangga yang belum disentuh user diberi skor berupa jumlah kemiripan tetangga yang menyentuhnya, lalu diurutkan dari skor tertinggi (seri diurutkan dari ID item terkecil). Ground truth adalah item yang disentuh user pada P2, belum pernah disentuh pada P1, dan ada di katalog P1.

Nomor 1a: skrip KNN. Fungsi cari_rekomendasi(u, k) pada notebook menghitung irisan item dengan perkalian matriks sparse (biner @ vektor_u.T), menghitung Jaccard, mengambil k tetangga terdekat, lalu memberi skor item kandidat.

Proses acak yang dapat direproduksi: calon user (3.905 user dengan minimal 3 item di P1 dan minimal 1 item ground truth) diurutkan berdasarkan ID, diacak dengan random.seed(123), lalu diambil user pertama yang punya minimal 3 kandidat rekomendasi dari 5 tetangganya. User terpilih: 204907.

Nomor 1b: tiga item yang diiklankan (k = 5).

Tetangga	Jaccard
237763	0,5
692676	0,4
832839	0,4
1243615	0,2857
60357	0,25
Rekomendasi top-5: 357044 (0,4), 448601 (0,4), 260611 (0,2857), 317435 (0,2857), 361253 (0,2857).
Tiga item yang diiklankan: 357044, 448601, dan 260611.

Nomor 2: evaluasi. Ground truth user 204907 adalah item 353834, yang tidak muncul pada top-5. Maka Hit@5 = 0 dan NDCG@5 = 0.

Sebagai pelengkap, evaluasi pada 300 user acak (random.seed(321)): 161 user mendapat minimal satu rekomendasi, rata-rata Hit@5 sebesar 0,0267 dan rata-rata NDCG@5 sebesar 0,0108. Nilai rendah ini wajar karena matriks sangat jarang (sparsity 99,999 persen), sebagian besar user hanya menyentuh satu item, dan ground truth hanya beberapa item dari sekitar 150 ribu item.

Teknologi
Python 3.10, Neo4j (Desktop) dan Python driver, Neo4j Graph Data Science, pandas, NumPy, SciPy (matriks sparse), NetworkX, Matplotlib, Jupyter Notebook.

Sumber Data
RetailRocket Recommender System Dataset, Roman Zykov, Kaggle: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
