#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

int main()
{
    CvSeqWriter writer;
    CvMemStorage * storage = cvCreateMemStorage(0);
    cvStartWriteSeq(CV_8UC1, sizeof(CvSeq), sizeof(CvPoint), storage, &writer);
    for (int i = 1; i <= 10; i++)
    {
        CvPoint pt;
        pt.x = 2 * i - 1;
        pt.y = 2 * i;
        CV_WRITE_SEQ_ELEM(pt, writer);
    }
    CvSeq* seq = cvEndWriteSeq(&writer);

    CvPoint temp;
    CvSeqReader reader;
    cvStartReadSeq(seq, &reader, 0);
    for (int i = 0; i<seq->total; i++)
    {
        CV_READ_SEQ_ELEM(temp, reader);
        cout << temp.x << " " << temp.y << endl;
    }
    cout << "\n" << endl;

    cvStartReadSeq(seq, &reader, 1);
    for (int i = 0; i<seq->total; i++)
    {
        CV_READ_SEQ_ELEM(temp, reader);
        cout << temp.x << " " << temp.y << endl;
    }
    return 0;
}
