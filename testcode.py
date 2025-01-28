from pyvi import ViTokenizer, ViPosTagger

print(ViTokenizer.tokenize(u"Phát biểu tại phiên thảo luận về tình hình kinh tế xã hội của Quốc hội sáng 28/10 , Bộ trưởng Bộ LĐ-TB&XH Đào Ngọc Dung khái quát , tại phiên khai mạc kỳ họp , lãnh đạo chính phủ đã báo cáo , đề cập tương đối rõ ràng về việc thực hiện các chính sách an sinh xã hội"))

ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội"))

from pyvi import ViUtils
ViUtils.remove_accents(u"Trường đại học bách khoa hà nội")

from pyvi import ViUtils
ViUtils.add_accents(u'truong dai hoc bach khoa ha noi')

from underthesea import word_tokenize
print(word_tokenize(u"Phát biểu tại phiên thảo luận về tình hình kinh tế xã hội của Quốc hội sáng 28/10 , Bộ trưởng Bộ LĐ-TB&XH Đào Ngọc Dung khái quát , tại phiên khai mạc kỳ họp , lãnh đạo chính phủ đã báo cáo , đề cập tương đối rõ ràng về việc thực hiện các chính sách an sinh xã hội", format="text"))