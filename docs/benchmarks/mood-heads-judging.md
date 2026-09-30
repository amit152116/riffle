# Mood-head judging sheet

Model: Essentia classification heads on Discogs-EffNet embeddings. 416 bench tracks.
Score = head's probability for the named class (0–1).
For each row put **Y** if the score matches what you hear, **N** if not, **?** if unsure.

Question per head: is the HIGH group really that mood, and the LOW group really not?

## mood_happy — class: happy

**HIGH (model says: yes)**

| score | track                                                                                                                 | your call |
| ----- | --------------------------------------------------------------------------------------------------------------------- | --------- |
| 0.96  | 600 - Main Koi Aisa Geet Gaoon - HD VIDEO ｜ Shah Rukh Khan & Juhi Chawla ｜ Yes Boss ｜ 90's Romantic Songs [yt]     | Y         |
| 0.95  | Paani_Wala_Dance(wapking.cc) [lib]                                                                                    | Y         |
| 0.94  | 184 - Dil Lutiya ｜ Jazzy B ｜ Ft. Apache Indian ｜ Sukshinder ｜ Romeo ｜ Jihne Mera Dil Luteya ｜ Punjabi Hits [yt] | Y         |
| 0.94  | 084 - Mittran Da Naa Chalda (Lyrical Video Song) Harjit Harman ｜ Atul Sharma ｜ Punjabi Song [yt]                    | Y         |
| 0.94  | 194 - MITTRAAN DA NAA CHALDA [yt]                                                                                     | Y         |
| 0.94  | 384 - Ambersar De Papad ｜ Gippy Grewal ｜ Sargun Mehta ｜ Sunidhi Chauhan ｜ Chandigarh Amritsar Chandigarh [yt]     | Y         |

**LOW (model says: no)**

| score | track                                                                                                                 | your call |
| ----- | --------------------------------------------------------------------------------------------------------------------- | --------- |
| 0.01  | 183 - Sapna Jahan (From ＂Brothers＂) [yt]                                                                            | N         |
| 0.01  | Sapna Jahan-192Kbps(SONGSPK.AUDIO) [lib]                                                                              | N         |
| 0.02  | 345 - Soulful Sufi Songs of Rahat Fateh Ali Khan ｜ AUDIO JUKEBOX ｜ Best of Rahat Fateh Ali Khan Songs [yt]          | N         |
| 0.02  | 312 - Heer - Full Song ｜ Jab Tak Hai Jaan ｜ Shah Rukh Khan ｜ Katrina Kaif ｜ Harshdeep Kaur ｜ A. R. Rahman [yt]   | N         |
| 0.02  | 256 - Shaabaashiyaan - Full Video ｜ Mission Mangal ｜ Akshay ｜ Vidya ｜ Sonakshi ｜ Taapsee [yt]                    | Y         |
| 0.03  | 181 - Naina - Lyrical ｜ Dangal ｜ Aamir Khan ｜ Arijit Singh ｜ Pritam ｜ Amitabh Bhattacharya ｜ New Song 2017 [yt] | N         |

## mood_sad — class: sad

**HIGH (model says: yes)**

| score | track                                                                                                              | your call |
| ----- | ------------------------------------------------------------------------------------------------------------------ | --------- |
| 0.94  | 019 - Dooron Dooron [yt]                                                                                           | Y         |
| 0.94  | 011 - Dooron Dooron (Official Video) - Paresh Pahuja Feat. Harleen Sethi ｜ Shiv ｜ Meghdeep ｜ Vaibhav [yt]       | Y         |
| 0.93  | 586 - Kho Gaye Hum Kahan -Full Video ｜Baar Baar Dekho ｜ Sidharth Malhotra, Katrina K｜ Jasleen R, Prateek K [yt] | Y         |
| 0.91  | 345 - Soulful Sufi Songs of Rahat Fateh Ali Khan ｜ AUDIO JUKEBOX ｜ Best of Rahat Fateh Ali Khan Songs [yt]       | Y         |
| 0.91  | Sapna Jahan-192Kbps(SONGSPK.AUDIO) [lib]                                                                           | Y         |
| 0.90  | Maa-(Mr-Jatt.com) [lib]                                                                                            | Y         |

**LOW (model says: no)**

| score | track                                                                                                    | your call |
| ----- | -------------------------------------------------------------------------------------------------------- | --------- |
| 0.02  | Birthday Bash - Yo Yo Honey Singh (Dilliwaali Zaalim Girlfriend) 320Kbps [lib]                           | N         |
| 0.02  | 605 - Koi Jaye To Le Aaye - Ghatak： Lethal (1996) Mamta Kulkarni ｜ Sunny Deol ｜ Full Video 1080p [yt] | N         |
| 0.02  | Insha Allah - www.downloadming.com [lib]                                                                 | N         |
| 0.02  | Paathshala - www.downloadming.com [lib]                                                                  | N         |
| 0.02  | Half Window Down Ft Neetu Singh - Ikka (DJJOhAL.Com) [lib]                                               | N         |
| 0.03  | 269 - EXPERT JATT - NAWAB (Official Video) Mista Baaz ｜ Juke Dock [yt]                                  | N         |

## mood_aggressive — class: aggressive

**HIGH (model says: yes)**

| score | track                                                                                      | your call |
| ----- | ------------------------------------------------------------------------------------------ | --------- |
| 0.73  | 03 Drama Queen [lib]                                                                       | N         |
| 0.60  | Birthday Bash - Yo Yo Honey Singh (Dilliwaali Zaalim Girlfriend) 320Kbps [lib]             | Y         |
| 0.53  | Paathshala - www.downloadming.com [lib]                                                    | Y         |
| 0.49  | Half Window Down Ft Neetu Singh - Ikka (DJJOhAL.Com) [lib]                                 | Y         |
| 0.47  | 328 - Half Window Down (Full Song) ｜ Ikka ｜ Dr Zeus ｜ Neetu Singh ｜ Speed Records [yt] | Y         |
| 0.43  | Botlan (DJJOhAL [lib]                                                                      | Y         |

**LOW (model says: no)**

| score | track                                                                                                            | your call |
| ----- | ---------------------------------------------------------------------------------------------------------------- | --------- |
| 0.00  | 011 - Dooron Dooron (Official Video) - Paresh Pahuja Feat. Harleen Sethi ｜ Shiv ｜ Meghdeep ｜ Vaibhav [yt]     | N         |
| 0.00  | 019 - Dooron Dooron [yt]                                                                                         | N         |
| 0.00  | 193 - Taarif Karoon Kya Uski with lyrics ｜ तारीफ़ करूं क्या उसकी के बोल ｜ Kashmir Ki Kali ｜ Shammi Kapoor [yt] | Y N       | N   |
| 0.00  | 090 - Tu Hain Toh Main Hoon [yt]                                                                                 | N         |
| 0.00  | 343 - Tum Itna Jo - Papon ｜ MTV Unplugged [yt]                                                                  | N         |
| 0.00  | 110 - Tu Har Lamha - Lyrical Video ｜ Khamoshiyan ｜ Arijit Singh ｜ Ali Fazal ｜ Sapna Pabbi [yt]               | N         |

## mood_relaxed — class: relaxed

**HIGH (model says: yes)**

| score | track                                                                                                        | your call |
| ----- | ------------------------------------------------------------------------------------------------------------ | --------- |
| 0.99  | 019 - Dooron Dooron [yt]                                                                                     | Y         |
| 0.99  | 183 - Sapna Jahan (From ＂Brothers＂) [yt]                                                                   | Y         |
| 0.99  | 011 - Dooron Dooron (Official Video) - Paresh Pahuja Feat. Harleen Sethi ｜ Shiv ｜ Meghdeep ｜ Vaibhav [yt] | Y         |
| 0.99  | Sapna Jahan-192Kbps(SONGSPK.AUDIO) [lib]                                                                     | Y         |
| 0.99  | 122 - Mehrama - Love Aaj Kal ｜ Kartik ｜ Sara ｜ Pritam ｜ Darshan Raval ｜ Antara [yt]                     | Y         |
| 0.99  | Naina - Dangal (Arijit Singh) 320Kbps [lib]                                                                  | Y         |

**LOW (model says: no)**

| score | track                                                                          | your call |
| ----- | ------------------------------------------------------------------------------ | --------- |
| 0.03  | Birthday Bash - Yo Yo Honey Singh (Dilliwaali Zaalim Girlfriend) 320Kbps [lib] | N         |
| 0.04  | Paathshala - www.downloadming.com [lib]                                        | N         |
| 0.05  | Botlan (DJJOhAL [lib]                                                          | N         |
| 0.06  | Insha Allah - www.downloadming.com [lib]                                       | N         |
| 0.06  | Bebo - Yo Yo Honey Singh (PagalWorld.com)- 320Kbps [lib]                       | N         |
| 0.07  | 03 Drama Queen [lib]                                                           | N         |

## mood_party — class: party

**HIGH (model says: yes)**

| score | track                                                                                                              | your call |
| ----- | ------------------------------------------------------------------------------------------------------------------ | --------- |
| 1.00  | Life - Akhil (DJJOhAL.Com) [lib]                                                                                   | Y         |
| 1.00  | 400 - Official Video： Nikle Currant Song ｜ Jassi Gill ｜ Neha Kakkar ｜ Sukh-E Muzical Doctorz ｜ Jaani [yt]     | Y         |
| 1.00  | 080 - Akhil Feat Adah Sharma ｜ Life Official Video ｜ Preet Hundal ｜ Arvindr Khaira ｜ Latest Punjabi Songs [yt] | Y         |
| 1.00  | Na_Na_Na_Na-J_Star[www.Mp3MaD.Com] [lib]                                                                           | Y         |
| 1.00  | Nikle Currant Ft Neha Kakkar - Jassi Gill (DJJOhAL.Com) [lib]                                                      | Y         |
| 1.00  | 269 - EXPERT JATT - NAWAB (Official Video) Mista Baaz ｜ Juke Dock [yt]                                            | Y         |

**LOW (model says: no)**

| score | track                                                                                                             | your call |
| ----- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| 0.00  | 002 - Navjot Ahuja - Khat X The Notebook [yt]                                                                     | N         |
| 0.00  | 008 - Navjot Ahuja - Khat (Lyrics) ｜ Tere liye mandir jaaun Tere naam ka dia jalaaun [yt]                        | N         |
| 0.00  | 019 - Dooron Dooron [yt]                                                                                          | N         |
| 0.00  | 011 - Dooron Dooron (Official Video) - Paresh Pahuja Feat. Harleen Sethi ｜ Shiv ｜ Meghdeep ｜ Vaibhav [yt]      | N         |
| 0.00  | 581 - Chal Ghar Chalen (From ＂Malang - Unleash The Madness＂) (feat. Arijit Singh) [yt]                          | N         |
| 0.00  | 323 - LYRICAL： Chal Ghar Chalen ｜ Malang ｜ Aditya R K, Disha P ｜ Mithoon ft. Arijit Singh, Sayeed Quadri [yt] | N         |

## danceability — class: danceable

**HIGH (model says: yes)**

| score | track                                                                                                              | your call |
| ----- | ------------------------------------------------------------------------------------------------------------------ | --------- |
| 1.00  | 080 - Akhil Feat Adah Sharma ｜ Life Official Video ｜ Preet Hundal ｜ Arvindr Khaira ｜ Latest Punjabi Songs [yt] | Y         |
| 1.00  | Life - Akhil (DJJOhAL.Com) [lib]                                                                                   | Y         |
| 1.00  | 269 - EXPERT JATT - NAWAB (Official Video) Mista Baaz ｜ Juke Dock [yt]                                            | Y         |
| 1.00  | 550 - Mehbooba Mehbooba ｜ Ajnabee ｜ Akshay Kumar, Bipasha Basu ｜ Adnan Sami, Sunidhi Chauhan ｜ Anu Malik [yt]  | Y         |
| 1.00  | 010 - Dupatta Mera (Video) Mujhe Kucch Kehna Hai ｜ Kareena Kapoor & Tusshar Kapoor ｜ Anuradha Sriram [yt]        | Y         |
| 1.00  | Paani_Wala_Dance(wapking.cc) [lib]                                                                                 | Y         |

**LOW (model says: no)**

| score | track                                                                                                             | your call |
| ----- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| 0.03  | 581 - Chal Ghar Chalen (From ＂Malang - Unleash The Madness＂) (feat. Arijit Singh) [yt]                          | N         |
| 0.03  | 008 - Navjot Ahuja - Khat (Lyrics) ｜ Tere liye mandir jaaun Tere naam ka dia jalaaun [yt]                        | N         |
| 0.03  | 002 - Navjot Ahuja - Khat X The Notebook [yt]                                                                     | N         |
| 0.03  | Pal Pal Har Pal - Sonu_Nigam-www.Mp3Mad.Com [lib]                                                                 | N         |
| 0.04  | Maa Tu Bataa-(Mr-Jatt.com) [lib]                                                                                  | N         |
| 0.04  | 323 - LYRICAL： Chal Ghar Chalen ｜ Malang ｜ Aditya R K, Disha P ｜ Mithoon ft. Arijit Singh, Sayeed Quadri [yt] | N         |

## voice_instrumental — class: has vocals (vs instrumental)

**HIGH (model says: yes)**

| score | track                                                                          | your call |
| ----- | ------------------------------------------------------------------------------ | --------- |
| 1.00  | 013 - Prabh Singh Ft Jay Trak - 9_45 (Official Music Video) [yt]               | Y         |
| 1.00  | Birthday Bash - Yo Yo Honey Singh (Dilliwaali Zaalim Girlfriend) 320Kbps [lib] | Y         |
| 1.00  | 027 - Dilawara [yt]                                                            | Y         |
| 1.00  | Jatt Da Muqabala - Sidhu Moose Wala (DJJOhAL.Com) [lib]                        | Y         |
| 1.00  | Flying Cars - Ninja (DJJOhAL.Com) [lib]                                        | Y         |
| 1.00  | Dj Waley Babu - Badshah [lib]                                                  | Y         |

**LOW (model says: no)**

| score | track                                                                                                              | your call |
| ----- | ------------------------------------------------------------------------------------------------------------------ | --------- |
| 0.03  | 095 - RAAG YAMAN - RAAGEET LIVE ｜｜ PRIYANI VANI PANDITT [yt]                                                     | N         |
| 0.39  | 294 - Main Mast [yt]                                                                                               | Y         |
| 0.50  | Manali Trance - Yo Yo Honey Singh (The Shaukeen) - 320kbps [lib]                                                   | N         |
| 0.52  | 266 - Bhare Naina ｜Pratibha Singh Baghel,Deepak Pandit,Paras Nath ｜Vishal Shekhar｜Classical Hindi Sad Song [yt] | Y         |
| 0.53  | 299 - O Sanam [yt]                                                                                                 | Y         |
| 0.56  | Dil-De-Diya-Hai- [lib]                                                                                             | Y         |

## tonal_atonal — class: tonal

**HIGH (model says: yes)**

| score | track                                                                                                             | your call |
| ----- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| 0.93  | 093 - Dil Se Dil [yt]                                                                                             |           |
| 0.91  | 343 - Tum Itna Jo - Papon ｜ MTV Unplugged [yt]                                                                   |           |
| 0.87  | Jo Bejhi Thi Dua - Shanghai 320Kbps [lib]                                                                         |           |
| 0.86  | 019 - Dooron Dooron [yt]                                                                                          |           |
| 0.86  | 323 - LYRICAL： Chal Ghar Chalen ｜ Malang ｜ Aditya R K, Disha P ｜ Mithoon ft. Arijit Singh, Sayeed Quadri [yt] |           |
| 0.86  | Roke Na Ruke Naina - Badrinath Ki Dulhania [lib]                                                                  |           |

**LOW (model says: no)**

| score | track                                                                                                            | your call |
| ----- | ---------------------------------------------------------------------------------------------------------------- | --------- |
| 0.00  | 335 - Meri Bheegi Bheegi Si with lyrics ｜ मेरी भीगी भीगी सी के बोल ｜ Kishore Kumar [yt]                        |
| 0.00  | 193 - Taarif Karoon Kya Uski with lyrics ｜ तारीफ़ करूं क्या उसकी के बोल ｜ Kashmir Ki Kali ｜ Shammi Kapoor [yt] |
| 0.00  | Issey Kehte Hain Hip Hop (Yo Yo Honey Singh) -320Kbps [PagalWorld.com] [lib]                                     |           |
| 0.00  | 218 - Zihaale - E- Miskin (Original Song) Lata Mangeshkar, Shabbir Kumar ｜ Ghulami 1985 Songs ｜ Mithun [yt]    |           |
| 0.00  | Raat Jashan Di - Zorawar - Yo Yo Honey Singh 320Kbps [lib]                                                       |           |
| 0.00  | 416 - Gippy Grewal Feat Bohemia： Car Nachdi Official Video ｜ Jaani, B Praak ｜ Parul Yadav [yt]                |           |

## mood_acoustic — class: acoustic

**HIGH (model says: yes)**

| score | track                                                                                                              | your call |
| ----- | ------------------------------------------------------------------------------------------------------------------ | --------- |
| 0.98  | 215 - Sajni [yt]                                                                                                   | Y         |
| 0.98  | 011 - Dooron Dooron (Official Video) - Paresh Pahuja Feat. Harleen Sethi ｜ Shiv ｜ Meghdeep ｜ Vaibhav [yt]       | Y         |
| 0.98  | 581 - Chal Ghar Chalen (From ＂Malang - Unleash The Madness＂) (feat. Arijit Singh) [yt]                           | Y         |
| 0.98  | 323 - LYRICAL： Chal Ghar Chalen ｜ Malang ｜ Aditya R K, Disha P ｜ Mithoon ft. Arijit Singh, Sayeed Quadri [yt]  | Y         |
| 0.98  | 586 - Kho Gaye Hum Kahan -Full Video ｜Baar Baar Dekho ｜ Sidharth Malhotra, Katrina K｜ Jasleen R, Prateek K [yt] | Y         |
| 0.98  | 008 - Navjot Ahuja - Khat (Lyrics) ｜ Tere liye mandir jaaun Tere naam ka dia jalaaun [yt]                         | Y         |

**LOW (model says: no)**

| score | track                                                                                                             | your call |
| ----- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| 0.00  | Paathshala - www.downloadming.com [lib]                                                                           | N         |
| 0.00  | 010 - Dupatta Mera (Video) Mujhe Kucch Kehna Hai ｜ Kareena Kapoor & Tusshar Kapoor ｜ Anuradha Sriram [yt]       | N         |
| 0.00  | Birthday Bash - Yo Yo Honey Singh (Dilliwaali Zaalim Girlfriend) 320Kbps [lib]                                    | N         |
| 0.00  | 605 - Koi Jaye To Le Aaye - Ghatak： Lethal (1996) Mamta Kulkarni ｜ Sunny Deol ｜ Full Video 1080p [yt]          | N         |
| 0.00  | 550 - Mehbooba Mehbooba ｜ Ajnabee ｜ Akshay Kumar, Bipasha Basu ｜ Adnan Sami, Sunidhi Chauhan ｜ Anu Malik [yt] | N         |
| 0.00  | Insha Allah - www.downloadming.com [lib]                                                                          | N         |

## mood_electronic — class: electronic

**HIGH (model says: yes)**

| score | track                                                                                                    | your call |
| ----- | -------------------------------------------------------------------------------------------------------- | --------- |
| 1.00  | 605 - Koi Jaye To Le Aaye - Ghatak： Lethal (1996) Mamta Kulkarni ｜ Sunny Deol ｜ Full Video 1080p [yt] | Y         |
| 1.00  | Jhoom Jhoom (Papuyaar.com) [lib]                                                                         | Y         |
| 0.99  | 301 - Jhoom Jhoom (Full Song) ｜ Tom Dick And Harry [yt]                                                 | Y         |
| 0.99  | Aankhon Aankhon (Bhaag Johnny) Yo Yo Honey Singh 320Kbps [lib]                                           | Y         |
| 0.99  | Insha Allah - www.downloadming.com [lib]                                                                 | Y         |
| 0.99  | Aao Raja (Gabbar Is Back) Yo Yo Honey Singh 320Kbps [lib]                                                | Y         |

**LOW (model says: no)**

| score | track                                                                                                             | your call |
| ----- | ----------------------------------------------------------------------------------------------------------------- | --------- |
| 0.00  | 110 - Tu Har Lamha - Lyrical Video ｜ Khamoshiyan ｜ Arijit Singh ｜ Ali Fazal ｜ Sapna Pabbi [yt]                | N         |
| 0.00  | 002 - Navjot Ahuja - Khat X The Notebook [yt]                                                                     | N         |
| 0.00  | 008 - Navjot Ahuja - Khat (Lyrics) ｜ Tere liye mandir jaaun Tere naam ka dia jalaaun [yt]                        | N         |
| 0.00  | 090 - Tu Hain Toh Main Hoon [yt]                                                                                  | N         |
| 0.01  | 472 - Dil Tutda ｜ Jassi Gill ｜ Latest Punjabi Song 2017 ｜ Arvindr Khaira ｜ Goldboy ｜ Nirmaan [yt]            | N         |
| 0.01  | 464 - Raabta (Kehte Hain Khuda) Full Song With Lyrics ｜ Agent Vinod ｜ Saif Ali Khan, Kareena Kapoor,Pritam [yt] | N         |
