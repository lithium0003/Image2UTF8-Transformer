#include <stdio.h>
#include <sstream>
#include <ft2build.h>
#include FT_FREETYPE_H


int main(int argc, char *argv[])
{
	FT_Library  library;
	FT_Face     face;
	FT_Error    error;

	if(argc < 2) {
		fprintf(stderr, "Usage: %s font_path size\n",argv[0]);
		return 0;
	}

	error = FT_Init_FreeType(&library);
	if(error) {
		fprintf(stderr, "FT_Init_FreeType error %d\n", error);
		return 1;
	}

	error = FT_New_Face(library, argv[1], 0, &face);
	if(error) {
		fprintf(stderr, "FT_New_Face error %d\n", error);
		return 1;
	}

	double  size_f;
	std::istringstream(std::string(argv[2])) >> size_f;
	FT_F26Dot6 size = size_f*64;

	error = FT_Set_Char_Size(
          face,    /* handle to face object           */
          0,       /* char_width in 1/64th of points  */
          size,    /* char_height in 1/64th of points */
          72,      /* horizontal device resolution    */
          72 );    /* vertical device resolution      */
	if(error) {
		fprintf(stderr, "FT_Set_Char_Size error %d\n", error);
		return 1;
	}

	FT_ULong  charcode = 0;
	while(fread(&charcode, 4, 1, stdin) == 1) {
		FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
		if(glyph_index == 0) {
			unsigned long rows = 0;
			unsigned long width = 0;
			long bitmap_left = 0;
			long bitmap_top = 0;
			long advance_x = 0;
			long advance_y = 0;
			fwrite(&rows, sizeof(unsigned long), 1, stdout);
			fwrite(&width, sizeof(unsigned long), 1, stdout);
			fwrite(&bitmap_left, sizeof(long), 1, stdout);
			fwrite(&bitmap_top, sizeof(long), 1, stdout);
			fwrite(&advance_x, sizeof(long), 1, stdout);
			fwrite(&advance_y, sizeof(long), 1, stdout);
			fflush(stdout);
			continue;
		}

		error = FT_Load_Glyph(
		  face,          /* handle to face object */
		  glyph_index,   /* glyph index           */
		  FT_LOAD_DEFAULT);  /* load flags, see below */
		if(error) {
			fprintf(stderr, "FT_Load_Glyph error %d\n", error);
			return 1;
		}

		error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
		if(error) {
			fprintf(stderr, "FT_Render_Glyph error %d\n", error);
			return 1;
		}

		FT_GlyphSlot  slot = face->glyph;
		unsigned long rows = slot->bitmap.rows;
		unsigned long width = slot->bitmap.width;
		long bitmap_left = slot->bitmap_left;
		long bitmap_top = slot->bitmap_top;
		long advance_x = slot->advance.x;
		long advance_y = slot->advance.y;
		fwrite(&rows, sizeof(unsigned long), 1, stdout);
		fwrite(&width, sizeof(unsigned long), 1, stdout);
		fwrite(&bitmap_left, sizeof(long), 1, stdout);
		fwrite(&bitmap_top, sizeof(long), 1, stdout);
		fwrite(&advance_x, sizeof(long), 1, stdout);
		fwrite(&advance_y, sizeof(long), 1, stdout);
		fwrite(slot->bitmap.buffer, sizeof(char), rows*width, stdout);
		fflush(stdout);
	}

	return 0;	
}
