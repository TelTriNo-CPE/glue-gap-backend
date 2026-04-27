import { Controller, Get, Head, NotFoundException, Req, Res } from '@nestjs/common';
import { Request, Response } from 'express';
import { TilesService } from './tiles.service';

@Controller('tiles')
export class TilesController {
  constructor(private readonly tilesService: TilesService) {}

  @Get()
  async listTileSets() {
    const stems = await this.tilesService.listTileSets();
    return { tiles: stems };
  }

  @Get('*')
  @Head('*')
  async proxy(@Req() req: Request, @Res() res: Response) {
    // req.path = "/tiles/stem/stem.dzi" → strip leading "/" → "tiles/stem/stem.dzi"
    const minioKey = req.path.replace(/^\//, '');

    let body: NodeJS.ReadableStream;
    let contentType: string;
    try {
      ({ body, contentType } = await this.tilesService.getObject(minioKey));
    } catch (err) {
      if (err instanceof NotFoundException) {
        res.status(404).json({ message: err.message });
        return;
      }
      throw err;
    }

    res.setHeader('Content-Type', contentType);
    res.setHeader('Access-Control-Allow-Origin', '*');
    body.on('error', () => res.destroy());
    body.pipe(res);
  }
}
