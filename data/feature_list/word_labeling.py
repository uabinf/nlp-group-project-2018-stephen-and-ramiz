def getWords(url, x_path):
    import requests
    from lxml import html
    r = requests.get(url)
    assert r.status_code == 200
    text = r.text
    page = html.fromstring(text)
    myWords = page.xpath(x_path, smart_strings=False)
    return myWords


def writeToFile(filename, words, delimeter='\n'):
    with open(filename, 'w') as File:
        for word in words:
            File.write(word+delimeter)
def main(file, url):
    word_list = getWords(url, './/div[@class="wordlist-item"]/text()')
    writeToFile(file, word_list)

main('positiveWords.txt', 'https://www.enchantedlearning.com/wordlist/negativewords.shtml')
main('negativeWords.txt', 'https://www.enchantedlearning.com/wordlist/positivewords.shtml')