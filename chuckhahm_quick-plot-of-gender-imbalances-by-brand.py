#!/usr/bin/env python
# coding: utf-8



# Quick plot of gender imbalances by brand

library(readr)
library(dplyr)
library(ggplot2)
library(magrittr)


train <- read_csv('../input/gender_age_train.csv')
pbdm <- read_csv('../input/phone_brand_device_model.csv')
trans <- structure(list(phone_brand = c("<U+4E09><U+661F>", "<U+5929><U+8BED>", 
                                        "<U+6D77><U+4FE1>", "<U+8054><U+60F3>", "<U+6B27><U+6BD4>", "<U+7231><U+6D3E><U+5C14>", 
                                        "<U+52AA><U+6BD4><U+4E9A>", "<U+4F18><U+7C73>", "<U+6735><U+552F>", 
                                        "<U+9ED1><U+7C73>", "<U+9524><U+5B50>", "<U+9177><U+6BD4><U+9B54><U+65B9>", 
                                        "<U+7F8E><U+56FE>", "<U+5C3C><U+6BD4><U+9C81>", "<U+4E00><U+52A0>", 
                                        "<U+4F18><U+8D2D>", "<U+8BFA><U+57FA><U+4E9A>", "<U+7CD6><U+846B><U+82A6>", 
                                        "<U+4E2D><U+56FD><U+79FB><U+52A8>", "<U+8BED><U+4FE1>", "<U+57FA><U+4F0D>", 
                                        "<U+9752><U+6A59>", "<U+534E><U+7855>", "<U+590F><U+65B0>", "<U+7EF4><U+56FE>", 
                                        "<U+827E><U+4F18><U+5C3C>", "<U+6469><U+6258><U+7F57><U+62C9>", 
                                        "<U+4E61><U+7C73>", "<U+7C73><U+5947>", "<U+5927><U+53EF><U+4E50>", 
                                        "<U+6C83><U+666E><U+4E30>", "<U+795E><U+821F>", "<U+6469><U+4E50>", 
                                        "<U+98DE><U+79D2>", "<U+7C73><U+6B4C>", "<U+5BCC><U+53EF><U+89C6>", 
                                        "<U+5FB7><U+8D5B>", "<U+68A6><U+7C73>", "<U+4E50><U+89C6>", "<U+5C0F><U+6768><U+6811>", 
                                        "<U+7EBD><U+66FC>", "<U+90A6><U+534E>", "E<U+6D3E>", "<U+6613><U+6D3E>", 
                                        "<U+666E><U+8010><U+5C14>", "<U+6B27><U+65B0>", "<U+897F><U+7C73>", 
                                        "<U+6D77><U+5C14>", "<U+6CE2><U+5BFC>", "<U+7CEF><U+7C73>", "<U+552F><U+7C73>", 
                                        "<U+9177><U+73C0>", "<U+8C37><U+6B4C>", "<U+6602><U+8FBE>", "<U+8046><U+97F5>"
), english = c("samsung", "Ktouch", "hisense", "lenovo", "obi", 
               "ipair", "nubia", "youmi", "dowe", "heymi", "hammer", "koobee", 
               "meitu", "nibilu", "oneplus", "yougo", "nokia", "candy", "ccmc", 
               "yuxin", "kiwu", "greeno", "asus", "panosonic", "weitu", "aiyouni", 
               "moto", "xiangmi", "micky", "bigcola", "wpf", "hasse", "mole", 
               "fs", "mige", "fks", "desci", "mengmi", "lshi", "smallt", "newman", 
               "banghua", "epai", "epai", "pner", "ouxin", "ximi", "haier", 
               "bodao", "nuomi", "weimi", "kupo", "google", "ada", "lingyun"
)), class = c("tbl_df", "tbl", "data.frame"), row.names = c(NA, 
                                                            -55L), .Names = c("phone_brand", "english"))

train %<>% 
  left_join(pbdm) %>% 
  left_join(trans) %>% 
  mutate(english = ifelse(is.na(english), phone_brand, english))

head(train)
table(train$english) # Something weird happens here. This works on my machine.




# Quick plot of gender imbalances by brand

library(readr)
library(dplyr)
library(ggplot2)
library(magrittr)


train <- read_csv('../input/gender_age_train.csv')
pbdm <- read_csv('../input/phone_brand_device_model.csv')
trans <- structure(list(phone_brand = c("<U+4E09><U+661F>", "<U+5929><U+8BED>", 
                                        "<U+6D77><U+4FE1>", "<U+8054><U+60F3>", "<U+6B27><U+6BD4>", "<U+7231><U+6D3E><U+5C14>", 
                                        "<U+52AA><U+6BD4><U+4E9A>", "<U+4F18><U+7C73>", "<U+6735><U+552F>", 
                                        "<U+9ED1><U+7C73>", "<U+9524><U+5B50>", "<U+9177><U+6BD4><U+9B54><U+65B9>", 
                                        "<U+7F8E><U+56FE>", "<U+5C3C><U+6BD4><U+9C81>", "<U+4E00><U+52A0>", 
                                        "<U+4F18><U+8D2D>", "<U+8BFA><U+57FA><U+4E9A>", "<U+7CD6><U+846B><U+82A6>", 
                                        "<U+4E2D><U+56FD><U+79FB><U+52A8>", "<U+8BED><U+4FE1>", "<U+57FA><U+4F0D>", 
                                        "<U+9752><U+6A59>", "<U+534E><U+7855>", "<U+590F><U+65B0>", "<U+7EF4><U+56FE>", 
                                        "<U+827E><U+4F18><U+5C3C>", "<U+6469><U+6258><U+7F57><U+62C9>", 
                                        "<U+4E61><U+7C73>", "<U+7C73><U+5947>", "<U+5927><U+53EF><U+4E50>", 
                                        "<U+6C83><U+666E><U+4E30>", "<U+795E><U+821F>", "<U+6469><U+4E50>", 
                                        "<U+98DE><U+79D2>", "<U+7C73><U+6B4C>", "<U+5BCC><U+53EF><U+89C6>", 
                                        "<U+5FB7><U+8D5B>", "<U+68A6><U+7C73>", "<U+4E50><U+89C6>", "<U+5C0F><U+6768><U+6811>", 
                                        "<U+7EBD><U+66FC>", "<U+90A6><U+534E>", "E<U+6D3E>", "<U+6613><U+6D3E>", 
                                        "<U+666E><U+8010><U+5C14>", "<U+6B27><U+65B0>", "<U+897F><U+7C73>", 
                                        "<U+6D77><U+5C14>", "<U+6CE2><U+5BFC>", "<U+7CEF><U+7C73>", "<U+552F><U+7C73>", train %>%
  group_by(phone_brand) %>% 
  summarise(PERCENT_MALE = 100 *(mean(gender == 'M') - mean(train$gender == 'M')), n = n()) %>%
  filter(n > 30) %>% 
  arrange(PERCENT_MALE) %>% 
  mutate(phone_brand = factor(phone_brand, levels=phone_brand[order(PERCENT_MALE)])) %>% 
  ggplot(aes(x = phone_brand, y = PERCENT_MALE, fill=n)) +
    geom_bar(stat='identity') +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90)) +
    coord_flip() +
    scale_fill_continuous(trans='log') +
    ylab('Within-brand percent male - overall percent male') +
    xlab('Brand') +
    ggtitle('Gender biases in phone brands')




# Since it works on my computer I'll cheat a bit

structure(list(english = structure(1:40, .Label = c("meitu", 
"dowe", "vivo", "heymi", "OPPO", "mige", "koobee", "bodao", "samsung", 
"fks", "<U+9177><U+6BD4>", "<U+91D1><U+7ACB>", "youmi", "mengmi", 
"<U+5C0F><U+7C73>", "lenovo", "<U+7D22><U+5C3C>", "<U+534E><U+4E3A>", 
"HTC", "Ktouch", "yuxin", "<U+9177><U+6D3E>", "hisense", "LG", 
"<U+5EB7><U+4F73>", "<U+9B45><U+65CF>", "ccmc", "hammer", "<U+4E2D><U+5174>", 
"aiyouni", "lshi", "TCL", "asus", "nubia", "ipair", "moto", "lingyun", 
"oneplus", "<U+5947><U+9177>", "ZUK"), class = "factor"), PERCENT_MALE = c(-0.326967261462357, 
-0.314118237493985, -0.110834692135815, -0.0998995922894248, 
-0.0911746652718148, -0.0839332057348029, -0.0802567351465676, 
-0.0573908814880311, -0.0372308633805384, -0.0177567351465676, 
-0.0158910635047765, -0.00918907258066248, -0.00734006847990099, 
0.00430208838284418, 0.0154271213556746, 0.0271814691427817, 
0.0297264863299425, 0.0302451064740585, 0.0360543866175743, 0.0364885478723003, 
0.0454785589710794, 0.0479855979001507, 0.0484197354416677, 0.0690588268130866, 
0.0715289791391467, 0.0772268174850114, 0.0772432648534324, 0.0902275580471497, 
0.0912734622982639, 0.107243264853432, 0.107866706249941, 0.109243264853432, 
0.119955129260212, 0.12832760220283, 0.142957550567718, 0.182485983300034, 
0.185814693424861, 0.196323724623547, 0.200100407710575, 0.214386121996289
), n = c(57L, 213L, 5952L, 35L, 6068L, 34L, 64L, 41L, 14224L, 
40L, 67L, 1138L, 192L, 34L, 17840L, 2751L, 745L, 13575L, 1043L, 
159L, 170L, 3489L, 204L, 347L, 42L, 4864L, 275L, 191L, 861L, 
40L, 802L, 250L, 59L, 498L, 42L, 103L, 35L, 174L, 140L, 56L)), .Names = c("english", 
"PERCENT_MALE", "n"), class = c("tbl_df", "tbl", "data.frame"
), row.names = c(NA, -40L)) %>%
  mutate(PERCENT_MALE = 100 * PERCENT_MALE) %>%
  ggplot(aes(x = english, y = PERCENT_MALE, fill=n)) +
    geom_bar(stat='identity') +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90)) +
    coord_flip() +
    scale_fill_continuous(trans='log') +
    ylab('Within-brand percent male - overall percent male') +
    xlab('Brand') +
    ggtitle('Gender biases in phone brands')






